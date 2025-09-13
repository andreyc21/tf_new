import os
import time
import signal
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pybit.unified_trading import WebSocket, HTTP
import uuid
import logging
import threading
import requests
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from rsi_strategy import RSIStrategyBase
from config import USE_CUSTOM_RSI, USE_DUAL_RSI, USE_NEURAL_FILTER, NEURAL_CONFIDENCE_THRESHOLD

# === ЛОГГЕР ===
def setup_logging():
    """Настройка логирования в файл с ротацией"""
    import logging.handlers
    
    # Создаем директорию для логов
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Основной логгер
    logger = logging.getLogger("rsi-bot")
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Ротируемый файловый handler (100MB, 10 файлов)
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'bybit_bot.log'),
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Консольный handler (только для разработки)
    if os.getenv('DEVELOPMENT', 'false').lower() == 'true':
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Отдельный файл для ошибок
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'bybit_bot_errors.log'),
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logging()

# === HEALTH CHECK SERVER ===
class HealthCheckHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Отключаем стандартные логи HTTP сервера
        pass
    
    def do_GET(self):
        if self.path == '/health':
            try:
                # Проверяем состояние бота
                if global_bot_instance and hasattr(global_bot_instance, 'ws') and global_bot_instance.ws:
                    status = {
                        'status': 'healthy',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'connected': global_bot_instance.is_connected,
                        'last_tick': global_bot_instance.last_tick_time.isoformat() if global_bot_instance.last_tick_time else None,
                        'reconnect_attempts': global_bot_instance.reconnect_attempts,
                        'equity': global_bot_instance.strategy.equity,
                        'trades_count': len(global_bot_instance.strategy.trades)
                    }
                    self.send_response(200)
                else:
                    status = {
                        'status': 'unhealthy',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'error': 'Bot not initialized'
                    }
                    self.send_response(503)
                
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status, indent=2).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    'status': 'error',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'error': str(e)
                }
                self.wfile.write(json.dumps(error_response, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    """Запускает HTTP сервер для health check"""
    try:
        server = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
        server.timeout = 1  # Короткий таймаут для неблокирующей работы
        
        def serve_forever():
            while True:
                server.handle_request()
        
        health_thread = threading.Thread(target=serve_forever, daemon=True)
        health_thread.start()
        logger.info("🏥 Health check server запущен на порту 8080")
        return server
    except Exception as e:
        logger.error(f"❌ Не удалось запустить health check server: {e}")
        return None

# === НАСТРОЙКИ ===
API_KEY  = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
SYMBOL = 'BTCUSDT'
POSITION_SIZE = 0.01  # фиксированный размер позиции (например, 0.01 BTC)
RSI_PERIOD = 14
RSI_BUY = 30
RSI_SELL = 70
CANDLE_MINUTES = 5
# Определяем testnet по переменной окружения
TESTNET = os.environ.get('TESTNET', '1') == '1'

# === НАСТРОЙКИ TELEGRAM ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

# === ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ ДЛЯ ДАМПА ===
global_bot_instance = None

def create_debug_dump(bot_instance, signal_name="MANUAL"):
    """Создает дебаг-дамп состояния бота"""
    timestamp = datetime.now(timezone.utc)
    filename = f"debug_dump_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        # Собираем все данные о состоянии бота
        dump_data = {
            "timestamp": timestamp.isoformat(),
            "signal_name": signal_name,
            "bot_state": {
                "symbol": bot_instance.symbol,
                "position": bot_instance.position,
                "last_signal": bot_instance.last_signal,
                "position_size": bot_instance.position_size,
                "last_rsi": bot_instance.last_rsi,
                "is_connected": bot_instance.is_connected,
                "last_tick_time": bot_instance.last_tick_time
            },
            "strategy_state": {
                "position": bot_instance.strategy.position,
                "rsi_period": bot_instance.strategy.rsi_period,
                "rsi_buy": bot_instance.strategy.rsi_buy,
                "rsi_sell": bot_instance.strategy.rsi_sell,
                "bb_period": bot_instance.strategy.bb_period,
                "equity": bot_instance.strategy.equity,
                "total_candles": len(bot_instance.strategy.candles),
                "total_rsi_values": len(bot_instance.strategy.rsi_values),
                "total_trades": len(bot_instance.strategy.trades)
            },
            "recent_candles": [],
            "rsi_values": [],
            "bb_values": [],
            "entry_points": [],
            "exit_points": [],
            "equity_curve": []
        }
        
        # Добавляем последние 50 свечей
        recent_candles = bot_instance.strategy.candles[-50:] if len(bot_instance.strategy.candles) > 50 else bot_instance.strategy.candles
        for candle in recent_candles:
            dump_data["recent_candles"].append({
                "start_time": candle.start_time.isoformat(),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume
            })
        
        # Добавляем текущую свечу, если есть
        if bot_instance.strategy.current_candle:
            current_candle = {
                "start_time": bot_instance.strategy.current_candle.start_time.isoformat(),
                "open": bot_instance.strategy.current_candle.open,
                "high": bot_instance.strategy.current_candle.high,
                "low": bot_instance.strategy.current_candle.low,
                "close": bot_instance.strategy.current_candle.close,
                "volume": bot_instance.strategy.current_candle.volume,
                "is_current": True
            }
            dump_data["current_candle"] = current_candle
        
        # Добавляем последние 100 значений RSI
        dump_data["rsi_values"] = bot_instance.strategy.rsi_values[-100:] if len(bot_instance.strategy.rsi_values) > 100 else bot_instance.strategy.rsi_values
        
        # Добавляем последние 100 значений BB
        recent_bb = bot_instance.strategy.bb_values[-100:] if len(bot_instance.strategy.bb_values) > 100 else bot_instance.strategy.bb_values
        for bb in recent_bb:
            if bb[0] is not None:  # ma, upper, lower
                dump_data["bb_values"].append({
                    "ma": bb[0],
                    "upper": bb[1], 
                    "lower": bb[2]
                })
            else:
                dump_data["bb_values"].append(None)
        
        # Добавляем точки входа и выхода
        for entry_time, entry_price in bot_instance.strategy.entry_points:
            dump_data["entry_points"].append({
                "time": entry_time.isoformat(),
                "price": entry_price
            })
        
        for exit_time, exit_price in bot_instance.strategy.exit_points:
            dump_data["exit_points"].append({
                "time": exit_time.isoformat(), 
                "price": exit_price
            })
        
        # Добавляем кривую эквити
        dump_data["equity_curve"] = bot_instance.strategy.equity_curve[-200:] if len(bot_instance.strategy.equity_curve) > 200 else bot_instance.strategy.equity_curve
        
        # Рассчитываем текущий RSI вручную для проверки
        if len(bot_instance.strategy.candles) > 0:
            closes = [c.close for c in bot_instance.strategy.candles]
            if bot_instance.strategy.current_candle:
                closes.append(bot_instance.strategy.current_candle.close)
            
            from rsi_strategy import compute_rsi
            manual_rsi = compute_rsi(closes, bot_instance.strategy.rsi_period)
            dump_data["manual_rsi_check"] = {
                "calculated_rsi": manual_rsi,
                "last_strategy_rsi": bot_instance.strategy.rsi_values[-1] if bot_instance.strategy.rsi_values else None,
                "closes_used": closes[-15:],  # последние 15 цен закрытия
                "total_closes": len(closes)
            }
        
        # Сохраняем в файл
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🔍 [DEBUG DUMP] Создан дебаг-дамп: {filename}")
        logger.info(f"📊 [DEBUG DUMP] Свечей: {len(dump_data['recent_candles'])}, RSI значений: {len(dump_data['rsi_values'])}")
        logger.info(f"💹 [DEBUG DUMP] Текущая позиция: {dump_data['bot_state']['position']}, Последний RSI: {dump_data['bot_state']['last_rsi']}")
        
        # Отправляем уведомление в Telegram, если настроено
        if bot_instance.notifications:
            telegram_msg = f"🔍 <b>Debug Dump создан</b>\n\n" \
                          f"📁 Файл: <code>{filename}</code>\n" \
                          f"📊 Свечей: {len(dump_data['recent_candles'])}\n" \
                          f"📈 RSI: {dump_data['bot_state']['last_rsi']:.2f}\n" \
                          f"💼 Позиция: {dump_data['bot_state']['position']}\n" \
                          f"⏰ {timestamp.strftime('%H:%M:%S UTC')}"
            bot_instance.notifications.telegram.send_message(telegram_msg)
        
        return filename
        
    except Exception as e:
        logger.error(f"❌ [DEBUG DUMP] Ошибка создания дампа: {e}")
        return None

def signal_handler(signum, frame):
    """Обработчик сигнала для создания дебаг-дампа"""
    signal_names = {
        signal.SIGUSR1: "SIGUSR1",  # Debug dump
        signal.SIGUSR2: "SIGUSR2",  # Force reconnect
        signal.SIGHUP: "SIGHUP",    # Reset reconnection counter
        signal.SIGTERM: "SIGTERM",
        signal.SIGINT: "SIGINT"
    }
    
    signal_name = signal_names.get(signum, f"SIGNAL_{signum}")
    
    if global_bot_instance:
        if signum == signal.SIGUSR1:
            # Debug dump
            logger.info(f"🔔 [SIGNAL] Получен сигнал {signal_name} - создаем дебаг-дамп")
            filename = create_debug_dump(global_bot_instance, signal_name)
            if filename:
                logger.info(f"✅ [SIGNAL] Дамп сохранен в {filename}")
            else:
                logger.error(f"❌ [SIGNAL] Не удалось создать дамп")
                
        elif signum == signal.SIGUSR2:
            # Force reconnect
            logger.info(f"🔔 [SIGNAL] Получен сигнал {signal_name} - принудительное переподключение")
            global_bot_instance.force_reconnect()
            
        elif signum == signal.SIGHUP:
            # Reset reconnection counter
            logger.info(f"🔔 [SIGNAL] Получен сигнал {signal_name} - сброс счетчика попыток")
            global_bot_instance.reset_reconnection_counter()
            
        else:
            # Other signals - create dump too
            logger.info(f"🔔 [SIGNAL] Получен сигнал {signal_name} - создаем дебаг-дамп")
            filename = create_debug_dump(global_bot_instance, signal_name)
            if filename:
                logger.info(f"✅ [SIGNAL] Дамп сохранен в {filename}")
    else:
        logger.warning(f"⚠️ [SIGNAL] Экземпляр бота не найден")
    
    # Если это SIGTERM или SIGINT, завершаем работу
    if signum in [signal.SIGTERM, signal.SIGINT]:
        logger.info(f"🛑 [SIGNAL] Завершение работы по сигналу {signal_name}")
        if global_bot_instance and global_bot_instance.notifications:
            global_bot_instance.notifications.notify_bot_stop(f"Сигнал {signal_name}")
        exit(0)

# === КЛАСС TELEGRAM ===
class TelegramNotifier:
    def __init__(self, bot_token, chat_id, logger):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logger
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage" if bot_token else None
        self.enabled = bool(bot_token and chat_id)
        
        if self.enabled:
            self.logger.info("📱 [TELEGRAM] Telegram уведомления включены")
        else:
            self.logger.warning("📱 [TELEGRAM] Telegram уведомления отключены (не указан токен или chat_id)")
    
    def send_message(self, text, parse_mode="HTML"):
        """Отправляет сообщение в Telegram"""
        if not self.enabled:
            return False
        
        try:
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(self.api_url, data=data, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                self.logger.error(f"[TELEGRAM] Ошибка отправки: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.error("[TELEGRAM] Таймаут при отправке сообщения")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[TELEGRAM] Ошибка сети: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[TELEGRAM] Неожиданная ошибка: {e}")
            return False
    
    def test_connection(self):
        """Тестирует подключение к Telegram"""
        if not self.enabled:
            return False
        
        test_message = "🔧 <b>Тест подключения</b>\n\nTelegram уведомления работают корректно!"
        return self.send_message(test_message)

# === КЛАСС УВЕДОМЛЕНИЙ ===
class NotificationManager:
    def __init__(self, logger):
        self.logger = logger
        self.last_connection_status = None
        self.last_notification_time = {}
        self.notification_cooldown = 60  # секунды между повторными уведомлениями
        # Инициализируем Telegram
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, logger)
    
    def _can_notify(self, notification_type):
        """Проверяет, можно ли отправить уведомление (защита от спама)"""
        now = time.time()
        last_time = self.last_notification_time.get(notification_type, 0)
        return now - last_time >= self.notification_cooldown
    
    def _update_notification_time(self, notification_type):
        """Обновляет время последнего уведомления"""
        self.last_notification_time[notification_type] = time.time()
    
    def notify_connection_lost(self):
        """Уведомление о разрыве соединения"""
        if self._can_notify('connection_lost'):
            message = "🔴 [CONNECTION] Соединение с WebSocket потеряно!"
            self.logger.error(message)
            
            # Отправляем в Telegram
            telegram_msg = f"🔴 <b>Разрыв соединения</b>\n\n❌ WebSocket соединение потеряно\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
            self.telegram.send_message(telegram_msg)
            
            self._update_notification_time('connection_lost')
    
    def notify_connection_restored(self):
        """Уведомление о восстановлении соединения"""
        if self.last_connection_status == False:  # только если было разорвано
            message = "🟢 [CONNECTION] Соединение с WebSocket восстановлено!"
            self.logger.info(message)
            
            # Отправляем в Telegram
            telegram_msg = f"🟢 <b>Соединение восстановлено</b>\n\n✅ WebSocket соединение восстановлено\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
            self.telegram.send_message(telegram_msg)
            
            self._update_notification_time('connection_restored')
    
    def notify_trade_entry(self, signal, price, rsi):
        """Уведомление о входе в сделку"""
        direction = "ЛОНГ" if signal == 1 else "ШОРТ"
        emoji = "📈" if signal == 1 else "📉"
        
        message = f"{emoji} [TRADE ENTRY] Вход в {direction} по цене {price} (RSI: {rsi:.2f})"
        self.logger.info(message)
        
        # Отправляем в Telegram
        telegram_msg = f"{emoji} <b>Вход в позицию</b>\n\n" \
                      f"📊 Направление: <b>{direction}</b>\n" \
                      f"💰 Цена входа: <code>{price}</code>\n" \
                      f"📊 RSI: <code>{rsi:.2f}</code>\n" \
                      f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        self.telegram.send_message(telegram_msg)
    
    def notify_trade_exit(self, signal, price, rsi, pnl=None):
        """Уведомление о выходе из сделки"""
        direction = "ЛОНГА" if signal == 1 else "ШОРТА"
        emoji = "💹" if pnl and pnl > 0 else "📉"
        
        pnl_str = f" PnL: {pnl:.4f}" if pnl is not None else ""
        message = f"{emoji} [TRADE EXIT] Выход из {direction} по цене {price} (RSI: {rsi:.2f}){pnl_str}"
        self.logger.info(message)
        
        # Отправляем в Telegram
        pnl_emoji = "💰" if pnl and pnl > 0 else "💸" if pnl and pnl < 0 else "⚖"
        pnl_text = f"\n{pnl_emoji} PnL: <code>{pnl:.4f}</code>" if pnl is not None else ""
        
        telegram_msg = f"{emoji} <b>Выход из позиции</b>\n\n" \
                      f"📊 Закрытие: <b>{direction}</b>\n" \
                      f"💰 Цена выхода: <code>{price}</code>\n" \
                      f"📊 RSI: <code>{rsi:.2f}</code>{pnl_text}\n" \
                      f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        self.telegram.send_message(telegram_msg)
    
    def notify_order_placed(self, side, qty, price, order_id):
        """Уведомление о размещении ордера"""
        message = f"📋 [ORDER] Размещен {side} ордер: {qty} по цене {price} (ID: {order_id})"
        self.logger.info(message)
        
        # Отправляем в Telegram
        telegram_msg = f"📋 <b>Ордер размещен</b>\n\n" \
                      f"📊 Тип: <b>{side}</b>\n" \
                      f"💹 Количество: <code>{qty}</code>\n" \
                      f"💰 Цена: <code>{price}</code>\n" \
                      f"🏷 ID: <code>{order_id}</code>\n" \
                      f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        self.telegram.send_message(telegram_msg)
    
    def notify_order_cancelled(self, order_id, reason=""):
        """Уведомление об отмене ордера"""
        reason_str = f" ({reason})" if reason else ""
        message = f"❌ [ORDER] Отменен ордер {order_id}{reason_str}"
        self.logger.info(message)
        
        # Отправляем в Telegram (только если есть причина)
        if reason:
            telegram_msg = f"❌ <b>Ордер отменен</b>\n\n" \
                          f"🏷 ID: <code>{order_id}</code>\n" \
                          f"📝 Причина: {reason}\n" \
                          f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
            self.telegram.send_message(telegram_msg)
    
    def notify_error(self, error_msg, error_type="GENERAL"):
        """Уведомление об ошибке"""
        if self._can_notify(f'error_{error_type}'):
            message = f"⚠️ [ERROR {error_type}] {error_msg}"
            self.logger.error(message)
            
            # Отправляем в Telegram (только критичные ошибки)
            if error_type in ['CRITICAL', 'TRADE', 'WEBSOCKET']:
                emoji = "🚨" if error_type == 'CRITICAL' else "⚠️"
                telegram_msg = f"{emoji} <b>Ошибка {error_type}</b>\n\n" \
                              f"📝 {error_msg}\n" \
                              f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
                self.telegram.send_message(telegram_msg)
            
            self._update_notification_time(f'error_{error_type}')
    
    def notify_bot_start(self, symbol, testnet_mode):
        """Уведомление о запуске бота"""
        mode = "TESTNET" if testnet_mode else "MAINNET"
        message = f"🚀 [BOT] Запуск торгового бота на {mode}"
        self.logger.info(message)
        
        # Отправляем в Telegram
        telegram_msg = f"🚀 <b>Запуск торгового бота</b>\n\n" \
                      f"📊 Символ: <b>{symbol}</b>\n" \
                      f"🌐 Режим: <b>{mode}</b>\n" \
                      f"📈 RSI период: <code>{RSI_PERIOD}</code>\n" \
                      f"📉 RSI покупка: <code>{RSI_BUY}</code>\n" \
                      f"📈 RSI продажа: <code>{RSI_SELL}</code>\n" \
                      f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        self.telegram.send_message(telegram_msg)
    
    def notify_bot_stop(self, reason=""):
        """Уведомление об остановке бота"""
        reason_str = f" ({reason})" if reason else ""
        message = f"🛑 [BOT] Остановка бота{reason_str}"
        self.logger.info(message)
        
        # Отправляем в Telegram
        telegram_msg = f"🛑 <b>Остановка бота</b>\n\n" \
                      f"📝 Причина: {reason if reason else 'Пользовательский запрос'}\n" \
                      f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        self.telegram.send_message(telegram_msg)
    
    def test_telegram(self):
        """Тестирует Telegram уведомления"""
        return self.telegram.test_connection()

    def set_connection_status(self, status):
        """Устанавливает статус соединения"""
        if self.last_connection_status != status:
            if status:
                self.notify_connection_restored()
            else:
                self.notify_connection_lost()
            self.last_connection_status = status

class RSIBot:
    def __init__(self, http, ws, symbol, position_size):
        self.http = http
        self.ws = ws
        self.symbol = symbol
        self.position_size = position_size
        self.testnet = TESTNET  # Сохраняем настройку testnet
        # Используем RSIStrategyBase с конфигурируемой AI-стратегией
        self.strategy = RSIStrategyBase(
            rsi_period=RSI_PERIOD,
            rsi_buy=RSI_BUY,
            rsi_sell=RSI_SELL,
            candle_minutes=CANDLE_MINUTES,
            use_custom_rsi=USE_CUSTOM_RSI,  # 🏆 Конфигурируется в config.py
            use_dual_rsi=USE_DUAL_RSI,
            use_neural_filter=USE_NEURAL_FILTER,  # 🧠 AI-фильтр
            neural_confidence_threshold=NEURAL_CONFIDENCE_THRESHOLD
        )
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.last_signal = 0
        self.last_order_time = None
        self.last_rsi_print_minute = None
        self.last_rsi = None
        # Система уведомлений
        self.notifications = NotificationManager(logger)
        # Мониторинг соединения
        self.last_tick_time = time.time()
        self.connection_timeout = 30  # секунды без тиков для определения разрыва
        self.is_connected = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 30  # секунды между попытками переподключения
        self.dns_check_delay = 60  # секунды между DNS проверками
        self.last_reconnect_attempt = 0  # время последней попытки восстановления
        # Запускаем мониторинг соединения
        self._start_connection_monitor()

    def _start_connection_monitor(self):
        """Запускает мониторинг состояния соединения в отдельном потоке"""
        def monitor():
            while True:
                time.sleep(10)  # проверяем каждые 10 секунд
                current_time = time.time()
                time_since_last_tick = current_time - self.last_tick_time
                
                if time_since_last_tick > self.connection_timeout:
                    if self.is_connected:
                        self.is_connected = False
                        self.notifications.set_connection_status(False)
                        logger.warning(f"⏰ [CONNECTION] Нет тиков уже {time_since_last_tick:.0f} сек")
                        
                        # Начинаем процедуру восстановления только если прошло достаточно времени
                        current_time = time.time()
                        if current_time - self.last_reconnect_attempt >= self.reconnect_delay:
                            self.last_reconnect_attempt = current_time
                            self._attempt_reconnection()
                        else:
                            remaining_time = self.reconnect_delay - (current_time - self.last_reconnect_attempt)
                            logger.info(f"⏳ [CONNECTION] Ожидание перед следующей попыткой: {remaining_time:.0f} сек")
                else:
                    if not self.is_connected:
                        self.is_connected = True
                        self.reconnect_attempts = 0  # сбрасываем счетчик попыток
                        self.notifications.set_connection_status(True)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _attempt_reconnection(self):
        """Пытается восстановить соединение"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"❌ [RECONNECT] Достигнуто максимальное количество попыток ({self.max_reconnect_attempts})")
            self.notifications.telegram.send_message(
                f"❌ <b>Превышено максимальное количество попыток</b>\n\n"
                f"Попыток восстановления: {self.max_reconnect_attempts}\n"
                f"Требуется ручное вмешательство"
            )
            return
        
        self.reconnect_attempts += 1
        logger.info(f"🔄 [RECONNECT] Попытка восстановления #{self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        # Диагностируем проблему
        issue_type = self.diagnose_connection_issues()
        
        if issue_type == "NO_INTERNET":
            logger.warning(f"🌐 [RECONNECT] Нет интернета, попытка #{self.reconnect_attempts}")
            try:
                self.notifications.telegram.send_message(
                    f"🌐 <b>Проблема с интернетом</b>\n\n"
                    f"Попытка #{self.reconnect_attempts}/{self.max_reconnect_attempts}\n"
                    f"Следующая проверка через {self.reconnect_delay} сек"
                )
            except:
                pass  # Игнорируем ошибки Telegram при проблемах с интернетом
            
        elif issue_type == "DNS_FAILURE":
            logger.warning(f"🔍 [RECONNECT] DNS не работает, попытка #{self.reconnect_attempts}")
            try:
                self.notifications.telegram.send_message(
                    f"🔍 <b>Проблема с DNS</b>\n\n"
                    f"Не удается разрешить stream.bybit.com\n"
                    f"Попытка #{self.reconnect_attempts}/{self.max_reconnect_attempts}\n"
                    f"Следующая проверка через {self.dns_check_delay} сек"
                )
            except:
                pass  # Игнорируем ошибки Telegram при DNS проблемах
            
        elif issue_type == "API_DOWN":
            logger.warning(f"🔗 [RECONNECT] Bybit API недоступен, попытка #{self.reconnect_attempts}")
            try:
                self.notifications.telegram.send_message(
                    f"🔗 <b>Bybit API недоступен</b>\n\n"
                    f"Попытка #{self.reconnect_attempts}/{self.max_reconnect_attempts}\n"
                    f"Следующая проверка через {self.reconnect_delay} сек"
                )
            except:
                pass  # Игнорируем ошибки Telegram при проблемах с API
            
        else:  # WEBSOCKET_ISSUE
            logger.info(f"🔌 [RECONNECT] Проблема с WebSocket, пересоздаем соединение...")
            if self.recreate_websocket():
                logger.info("✅ [RECONNECT] WebSocket успешно восстановлен")
                try:
                    self.notifications.telegram.send_message(
                        f"✅ <b>Соединение восстановлено</b>\n\n"
                        f"WebSocket пересоздан успешно\n"
                        f"Попытка #{self.reconnect_attempts}/{self.max_reconnect_attempts}"
                    )
                except:
                    pass
                return
            else:
                logger.error("❌ [RECONNECT] Не удалось пересоздать WebSocket")
        
        # НЕ блокируем поток - просто логируем
        logger.info(f"⏳ [RECONNECT] Следующая попытка через {self.reconnect_delay} сек...")

    def cancel_my_orders(self):
        try:
            open_orders = self.http.get_open_orders(
                category="linear",
                symbol=self.symbol
            )
            for order in open_orders['result']['list']:
                link_id = order.get('orderLinkId', '')
                if link_id.startswith('rsi-bot-'):
                    order_id = order['orderId']
                    self.http.cancel_order(
                        category="linear",
                        symbol=self.symbol,
                        orderId=order_id
                    )
                    logger.info(f"[CANCEL] Cancelled order {order_id} ({link_id})")
        except Exception as e:
            logger.warning(f"Не удалось отменить ордера: {e}")

    def on_tick(self, price, dt):
        # Обновляем время последнего тика для мониторинга соединения
        self.last_tick_time = time.time()
        
        # Используем стратегию для обработки тика
        signal = self.strategy.on_tick(price, dt)
        
        # Получаем текущие значения для логирования
        if self.strategy.rsi_values:
            self.last_rsi = self.strategy.rsi_values[-1]
        
        # Отладочная информация
        rsi_str = f"{self.last_rsi:.2f}" if self.last_rsi is not None else "N/A"
        logger.debug(f"[DEBUG] Signal={signal}, Bot.position={self.position}, Last_signal={self.last_signal}, RSI={rsi_str}")
        
        # Дополнительная отладка для понимания почему нет торговли
        if self.last_rsi is not None:
            if signal != self.position:
                if signal == self.last_signal:
                    logger.debug(f"[DEBUG] Сигнал {signal} уже был обработан (last_signal={self.last_signal})")
                else:
                    logger.debug(f"[DEBUG] Новый сигнал! {self.last_signal} → {signal}")
            else:
                logger.debug(f"[DEBUG] Сигнал {signal} = текущая позиция {self.position}, торговля не нужна")
        
        # --- Торговля ---
        if signal != self.position and signal != self.last_signal:
            # Уведомления о торговых сигналах
            if signal == 1 and self.position == 0:
                self.notifications.notify_trade_entry(signal, price, self.last_rsi)
            elif signal == -1 and self.position == 0:
                self.notifications.notify_trade_entry(signal, price, self.last_rsi)
            elif signal == 0 and self.position != 0:
                self.notifications.notify_trade_exit(self.position, price, self.last_rsi)
            
            logger.info(f"[TRADE TRIGGER] Signal changed: {self.last_signal} -> {signal}, Current position: {self.position}")
            self.cancel_my_orders()
            self.trade(signal, price)
            self.last_signal = signal
        
        # --- Вывод RSI, цены и BB раз в минуту ---
        current_minute = dt.replace(second=0, microsecond=0)
        if self.last_rsi_print_minute is None or current_minute > self.last_rsi_print_minute:
            bb_str = ""
            if self.strategy.bb_values and len(self.strategy.bb_values) > 0:
                bb = self.strategy.bb_values[-1]
                if bb is not None:
                    ma, upper, lower = bb
                    bb_str = f" BB: lower={lower:.2f} MA={ma:.2f} upper={upper:.2f}"
            # Определяем статус позиции
            position_status = {
                1: "ЛОНГ",
                -1: "ШОРТ", 
                0: "БЕЗ ПОЗИЦИИ"
            }.get(self.position, "НЕИЗВЕСТНО")
            
            # Определяем следующий возможный сигнал
            next_action = ""
            if self.position == 0:  # без позиции
                if self.last_rsi < RSI_BUY:
                    next_action = f" → Готов к ЛОНГУ (RSI < {RSI_BUY})"
                elif self.last_rsi > RSI_SELL:
                    next_action = f" → Готов к ШОРТУ (RSI > {RSI_SELL})"
                else:
                    next_action = f" → Ожидание (RSI между {RSI_BUY}-{RSI_SELL})"
            elif self.position == 1:  # в лонге
                if self.last_rsi > RSI_SELL:
                    next_action = f" → Готов к ВЫХОДУ из лонга (RSI > {RSI_SELL})"
                else:
                    next_action = f" → Удерживаем ЛОНГ (RSI < {RSI_SELL})"
            elif self.position == -1:  # в шорте
                if self.last_rsi < RSI_BUY:
                    next_action = f" → Готов к ВЫХОДУ из шорта (RSI < {RSI_BUY})"
                else:
                    next_action = f" → Удерживаем ШОРТ (RSI > {RSI_BUY})"
            
            logger.info(f"[STATUS] Позиция: {position_status} | RSI: {self.last_rsi:.2f} | Цена: {price}{next_action}")
            if bb_str:
                logger.info(f"[BB]{bb_str}")
            self.last_rsi_print_minute = current_minute
    
    def create_manual_dump(self):
        """Создает дебаг-дамп вручную"""
        return create_debug_dump(self, "MANUAL")
    
    def force_reconnect(self):
        """Принудительное переподключение WebSocket"""
        logger.info("🔄 [MANUAL] Принудительное переподключение...")
        self.reconnect_attempts = 0  # Сбрасываем счетчик
        self.last_reconnect_attempt = 0  # Сбрасываем таймер
        
        if self.recreate_websocket():
            logger.info("✅ [MANUAL] Принудительное переподключение успешно")
            try:
                self.notifications.telegram.send_message("✅ <b>Принудительное переподключение</b>\n\nWebSocket пересоздан вручную")
            except:
                logger.warning("⚠️ [MANUAL] Не удалось отправить уведомление в Telegram")
        else:
            logger.error("❌ [MANUAL] Принудительное переподключение неудачно")
            try:
                self.notifications.telegram.send_message("❌ <b>Ошибка переподключения</b>\n\nНе удалось пересоздать WebSocket")
            except:
                logger.warning("⚠️ [MANUAL] Не удалось отправить уведомление в Telegram")
    
    def reset_reconnection_counter(self):
        """Сбрасывает счетчик попыток восстановления"""
        logger.info("🔄 [MANUAL] Сброс счетчика попыток восстановления")
        self.reconnect_attempts = 0
        self.last_reconnect_attempt = 0
        try:
            self.notifications.telegram.send_message("🔄 <b>Счетчик сброшен</b>\n\nПопытки восстановления: 0/10")
        except:
            logger.warning("⚠️ [MANUAL] Не удалось отправить уведомление в Telegram")
    
    def check_dns_resolution(self, host="stream.bybit.com"):
        """Проверяет разрешение DNS для хоста"""
        try:
            socket.gethostbyname(host)
            return True
        except socket.gaierror:
            return False
    
    def check_internet_connectivity(self):
        """Проверяет доступность интернета через ping"""
        try:
            # Пингуем Google DNS
            result = subprocess.run(['ping', '-c', '1', '-W', '3', '8.8.8.8'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_bybit_api_health(self):
        """Проверяет доступность Bybit API"""
        try:
            response = requests.get('https://api.bybit.com/v5/market/time', timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def diagnose_connection_issues(self):
        """Диагностирует проблемы с соединением"""
        logger.info("🔍 [ДИАГНОСТИКА] Проверяем сетевые проблемы...")
        
        # 1. Проверяем интернет
        internet_ok = self.check_internet_connectivity()
        logger.info(f"🌐 [ДИАГНОСТИКА] Интернет: {'✅ OK' if internet_ok else '❌ НЕТ'}")
        
        # 2. Проверяем DNS
        dns_ok = self.check_dns_resolution()
        logger.info(f"🔍 [ДИАГНОСТИКА] DNS (stream.bybit.com): {'✅ OK' if dns_ok else '❌ НЕТ'}")
        
        # 3. Проверяем Bybit API
        api_ok = self.check_bybit_api_health()
        logger.info(f"🔗 [ДИАГНОСТИКА] Bybit API: {'✅ OK' if api_ok else '❌ НЕТ'}")
        
        # Определяем тип проблемы
        if not internet_ok:
            return "NO_INTERNET"
        elif not dns_ok:
            return "DNS_FAILURE" 
        elif not api_ok:
            return "API_DOWN"
        else:
            return "WEBSOCKET_ISSUE"
    
    def recreate_websocket(self):
        """Пересоздает WebSocket соединение"""
        try:
            logger.info("🔄 [RECONNECT] Пересоздаем WebSocket соединение...")
            
            # Отписываемся от старого соединения
            try:
                self.ws.unsubscribe(f"publicTrade.{self.symbol}")
            except:
                pass
            
            # Создаем новое соединение
            self.ws = WebSocket(
                testnet=self.testnet,
                channel_type="linear"
            )
            
            # Подписываемся заново
            self.ws.subscribe(
                topic=f"publicTrade.{self.symbol}",
                callback=self.on_tick
            )
            
            logger.info("✅ [RECONNECT] WebSocket пересоздан успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ [RECONNECT] Ошибка пересоздания WebSocket: {e}")
            return False

    def trade(self, signal, price):
        logger.info(f"[TRADE] {datetime.fromtimestamp(time.time(), timezone.utc)} Signal: {signal}, Price: {price}")
        try:
            order_link_id = f"rsi-bot-{uuid.uuid4()}"
            offset = 0.0001  # 0.01%
            # Закрыть противоположную позицию лимитным ордером
            if self.position == 1 and signal == -1:
                limit_price = round(price * (1 + offset), 2)
                response = self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Sell",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    reduceOnly=True,
                    orderLinkId=order_link_id
                )
                if response.get('retCode') == 0:
                    order_id = response['result']['orderId']
                    self.notifications.notify_order_placed("SELL (закрытие лонга)", self.position_size, limit_price, order_id)
                    
            elif self.position == -1 and signal == 1:
                limit_price = round(price * (1 - offset), 2)
                response = self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Buy",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    reduceOnly=True,
                    orderLinkId=order_link_id
                )
                if response.get('retCode') == 0:
                    order_id = response['result']['orderId']
                    self.notifications.notify_order_placed("BUY (закрытие шорта)", self.position_size, limit_price, order_id)
            
            # Открыть новую позицию лимитным ордером
            if signal == 1:
                limit_price = round(price * (1 - offset), 2)
                response = self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Buy",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    orderLinkId=order_link_id
                )
                if response.get('retCode') == 0:
                    order_id = response['result']['orderId']
                    self.notifications.notify_order_placed("BUY (открытие лонга)", self.position_size, limit_price, order_id)
                    
            elif signal == -1:
                limit_price = round(price * (1 + offset), 2)
                response = self.http.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side="Sell",
                    orderType="Limit",
                    qty=self.position_size,
                    price=limit_price,
                    timeInForce="PostOnly",
                    orderLinkId=order_link_id
                )
                if response.get('retCode') == 0:
                    order_id = response['result']['orderId']
                    self.notifications.notify_order_placed("SELL (открытие шорта)", self.position_size, limit_price, order_id)
            
            self.position = signal
        except Exception as e:
            self.notifications.notify_error(f"Ошибка при размещении ордера: {e}", "TRADE")

# === Основной запуск ===
def main():
    global global_bot_instance
    
    # Запускаем health check сервер
    health_server = start_health_server()
    
    # Регистрируем обработчики сигналов
    signal.signal(signal.SIGUSR1, signal_handler)  # kill -USR1 <pid>
    signal.signal(signal.SIGUSR2, signal_handler)  # kill -USR2 <pid>
    signal.signal(signal.SIGHUP, signal_handler)   # kill -HUP <pid>
    signal.signal(signal.SIGTERM, signal_handler)  # kill <pid>
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    
    logger.info("🔔 [SIGNALS] Зарегистрированы обработчики сигналов:")
    logger.info("   SIGUSR1 (kill -USR1 <pid>) - создать дебаг-дамп")
    logger.info("   SIGUSR2 (kill -USR2 <pid>) - принудительное переподключение")
    logger.info("   SIGHUP (kill -HUP <pid>) - сбросить счетчик попыток") 
    logger.info("   SIGTERM (kill <pid>) - дамп + завершение")
    logger.info("   SIGINT (Ctrl+C) - дамп + завершение")
    
    http = HTTP(
        testnet=TESTNET,
        api_key=API_KEY,
        api_secret=API_SECRET
    )
    # --- Загрузка истории свечей ---
    logger.info(f"Loading historical candles... (testnet={TESTNET})")
    klines = http.get_kline(
        category="linear",
        symbol=SYMBOL,
        interval="5",
        limit=50
    )
    # klines['result']['list'] — список свечей от новых к старым!
    raw_candles = klines['result']['list'][::-1]  # теперь от старых к новым
    preload = []
    for c in raw_candles:
        preload.append({
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'start_time': datetime.fromtimestamp(int(c[0]) / 1000, timezone.utc)
        })
    # --- Вывод баланса ---
    try:
        balance = http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        usdt_balance = balance['result']['list'][0]['totalEquity']
        logger.info(f"Current USDT balance: {usdt_balance}")
    except Exception as e:
        logger.warning(f"Не удалось получить баланс: {e}")
    # --- Получение текущей позиции ---
    try:
        positions = http.get_positions(
            category="linear",
            symbol=SYMBOL
        )
        pos_list = positions['result']['list']
        current_position = 0
        found_position = False
        for pos in pos_list:
            size = float(pos['size'])
            side = pos['side']
            if size > 0:
                found_position = True
                if side == 'Buy':
                    current_position = 1
                elif side == 'Sell':
                    current_position = -1
                logger.info(f"Open position detected: {side} size={size}")
        if not found_position:
            logger.info(f"No open positions detected.")
    except Exception as e:
        logger.warning(f"Не удалось получить позицию: {e}")
        current_position = 0
    
    # --- WebSocket ---
    ws = WebSocket(
        testnet=TESTNET,
        channel_type="linear"
    )
    bot = RSIBot(http, ws, SYMBOL, POSITION_SIZE)
    
    # Устанавливаем глобальную ссылку для обработчика сигналов
    global_bot_instance = bot
    logger.info(f"🤖 [BOT] Экземпляр бота зарегистрирован для дебаг-дампов (PID: {os.getpid()})")
    
    # Предзагружаем исторические данные в стратегию
    for candle in preload:
        # Конвертируем в формат, который ожидает стратегия
        # Стратегия ожидает datetime объект, а не timestamp
        bot.strategy.on_tick(candle['close'], candle['start_time'])
    
    # --- Вывод последнего RSI и цены ---
    if bot.strategy.rsi_values:
        last_rsi = bot.strategy.rsi_values[-1]
        last_price = preload[-1]['close'] if preload else 0
        last_time = preload[-1]['start_time'] if preload else datetime.now(timezone.utc)
        logger.info(f"Last candle: {last_time.strftime('%Y-%m-%d %H:%M:%S')} Close: {last_price} RSI({RSI_PERIOD}): {last_rsi:.2f}")
    
    # ВАЖНО: Синхронизируем позиции бота и стратегии
    bot.position = current_position
    bot.strategy.position = current_position  # Синхронизируем стратегию с реальной позицией
    bot.last_signal = current_position  # Устанавливаем last_signal равным текущей позиции
    
    logger.info(f"[INIT] Bot position: {bot.position}, Strategy position: {bot.strategy.position}, Last signal: {bot.last_signal}")
    
    # Тестируем Telegram уведомления
    if TELEGRAM_ENABLED:
        logger.info("📱 [TELEGRAM] Тестирование Telegram уведомлений...")
        if bot.notifications.test_telegram():
            logger.info("✅ [TELEGRAM] Telegram уведомления работают")
        else:
            logger.warning("⚠️ [TELEGRAM] Проблема с Telegram уведомлениями")
    
    # Уведомление о запуске бота
    bot.notifications.notify_bot_start(SYMBOL, TESTNET)
    logger.info("Bot started. Waiting for ticks...")
    
    def handle_message(msg):
        if 'data' in msg and isinstance(msg['data'], list):
            for trade in msg['data']:
                price = float(trade['p'])
                ts = int(trade['T'])
                # Конвертируем timestamp в datetime объект
                dt = datetime.fromtimestamp(ts / 1000, timezone.utc)
                bot.on_tick(price, dt)
    
    try:
        ws.trade_stream(
            symbol=SYMBOL,
            callback=handle_message
        )
        
        # Основной цикл с обработкой исключений
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 [BOT] Остановка бота по запросу пользователя")
        bot.notifications.notify_bot_stop("Пользовательский запрос")
    except Exception as e:
        bot.notifications.notify_error(f"Критическая ошибка WebSocket: {e}", "CRITICAL")
        bot.notifications.notify_bot_stop(f"Критическая ошибка: {e}")
        logger.error(f"[WS CRITICAL ERROR] {e}")
    finally:
        logger.info("🏁 [BOT] Завершение работы бота")

if __name__ == "__main__":
    main() 