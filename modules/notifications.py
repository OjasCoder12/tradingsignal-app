import os
from twilio.rest import Client
from typing import List, Dict

class NotificationManager:
    def __init__(self):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = os.environ.get("TWILIO_PHONE_NUMBER")
        self.to_number = os.environ.get("USER_PHONE_NUMBER")
        self.client = None if not all([self.account_sid, self.auth_token, self.from_number]) else Client(self.account_sid, self.auth_token)

    def format_stock_message(self, signals: List[Dict]) -> str:
        """Format stock signals into a readable message"""
        buy_stocks = []
        hold_stocks = []
        
        for signal in signals:
            symbol = signal['symbol']
            price = signal['current_price']
            pred_price = signal['predicted_price']
            currency = signal['trading_decision']['fundamentals']['currency']
            
            if signal['signal'] == 'BUY':
                buy_stocks.append(
                    f"{symbol}: Current {currency}{price:.2f}, "
                    f"Predicted {currency}{pred_price:.2f}, "
                    f"Qty: {signal['recommended_quantity']}"
                )
            elif signal['signal'] == 'HOLD':
                hold_stocks.append(f"{symbol}: {currency}{price:.2f}")

        message = "🚨 Stock Alert 🚨\n\n"
        
        if buy_stocks:
            message += "🟢 Recommended Buys:\n" + "\n".join(buy_stocks) + "\n\n"
        
        if hold_stocks:
            message += "⚪ Hold Positions:\n" + "\n".join(hold_stocks)
            
        return message

    def send_notification(self, signals: List[Dict]) -> None:
        """Send notification about stock signals"""
        if not self.client or not self.to_number:
            return
            
        try:
            message = self.format_stock_message(signals)
            if message:
                self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=self.to_number
                )
        except Exception as e:
            print(f"Error sending notification: {str(e)}")
