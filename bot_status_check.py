#!/usr/bin/env python3.12
"""
Утилита для проверки состояния торгового бота
Показывает текущие позиции, ордера и статус
"""

import os
import sys
from datetime import datetime, timezone

# Попробуем импортировать pybit, если доступен
try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    print("⚠️ pybit не установлен. Только проверка настроек.")

def check_environment():
    """Проверяет переменные окружения"""
    print("🔧 Проверка переменных окружения:")
    
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET") 
    testnet = os.environ.get('TESTNET', '1') == '1'
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")
    
    print(f"   API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"   API_SECRET: {'✅ Set' if api_secret else '❌ Not set'}")
    print(f"   TESTNET: {'✅ Yes' if testnet else '❌ No (MAINNET)'}")
    print(f"   TELEGRAM_BOT_TOKEN: {'✅ Set' if telegram_token else '❌ Not set'}")
    print(f"   TELEGRAM_CHAT_ID: {'✅ Set' if telegram_chat else '❌ Not set'}")
    
    return api_key and api_secret

def check_bot_status():
    """Проверяет статус бота через API"""
    if not PYBIT_AVAILABLE:
        return
        
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    testnet = os.environ.get('TESTNET', '1') == '1'
    
    if not (api_key and api_secret):
        print("❌ API ключи не настроены")
        return
        
    try:
        http = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        
        print(f"\n💰 Проверка баланса ({'TESTNET' if testnet else 'MAINNET'}):")
        
        # Баланс
        balance = http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        usdt_balance = balance['result']['list'][0]['totalEquity']
        print(f"   USDT баланс: {usdt_balance}")
        
        # Позиции
        print(f"\n📊 Проверка позиций:")
        positions = http.get_positions(category="linear", symbol="BTCUSDT")
        pos_list = positions['result']['list']
        
        found_position = False
        for pos in pos_list:
            size = float(pos['size'])
            if size > 0:
                found_position = True
                side = pos['side']
                entry_price = pos['avgPrice']
                unrealized_pnl = pos['unrealisedPnl']
                print(f"   📈 Позиция: {side} {size} BTCUSDT")
                print(f"   💰 Цена входа: {entry_price}")
                print(f"   📊 Нереализованный PnL: {unrealized_pnl}")
        
        if not found_position:
            print("   ✅ Нет открытых позиций")
            
        # Открытые ордера
        print(f"\n📋 Проверка ордеров:")
        orders = http.get_open_orders(category="linear", symbol="BTCUSDT")
        order_list = orders['result']['list']
        
        bot_orders = [o for o in order_list if o.get('orderLinkId', '').startswith('rsi-bot-')]
        
        if bot_orders:
            for order in bot_orders:
                side = order['side']
                qty = order['qty'] 
                price = order['price']
                order_id = order['orderId']
                print(f"   📋 Ордер: {side} {qty} по цене {price} (ID: {order_id})")
        else:
            print("   ✅ Нет открытых ордеров от бота")
            
    except Exception as e:
        print(f"❌ Ошибка API: {e}")

def main():
    print("🤖 Проверка статуса торгового бота")
    print("=" * 50)
    
    env_ok = check_environment()
    
    if env_ok and PYBIT_AVAILABLE:
        check_bot_status()
    elif not PYBIT_AVAILABLE:
        print("\n📦 Для полной проверки установите: pip install pybit")
    else:
        print("\n❌ Настройте переменные окружения для API")
        
    print("\n" + "=" * 50)
    print("✅ Проверка завершена")

if __name__ == "__main__":
    main()
