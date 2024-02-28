import os
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
from dotenv import load_dotenv

load_dotenv()

TICKER = "SPY"     # Set to company ticker code in string

# To be configured in env file
ALPACA_CREDS = { 
    "API_KEY": os.getenv("API_KEY"),
    "API_SECRET": os.getenv("API_SECRET"),
    "PAPER": True
}

class MLTrader(Strategy):
    """
    A machine learning-based trading strategy.

    Attributes:
        symbol (str): The symbol of the asset to trade.
        sleeptime (str): The sleep time between trading iterations.
        last_trade (str): The type of the last trade (buy or sell).
        cash_at_risk (float): The percentage of remaining balance that can be used for the next investment.
        api (REST): The REST API object for making API calls.
    """

    def initialize(self, symbol:str=TICKER, cash_at_risk:float=.5):
        """
        Initializes the MLTrader strategy.

        Args:
            symbol (str): The symbol of the asset to trade.
            cash_at_risk (float): The percentage of remaining balance that can be used for the next investment.
        """
        self.symbol = symbol
        self.sleeptime = "24h"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=os.getenv("BASE_URL"), key_id=os.getenv("API_KEY"), secret_key=os.getenv("API_SECRET"))

    def get_dates(self):
        """
        Retrieves today's date and the date of the day 3 days prior.

        Returns:
            today (str): Today's date in the format 'YYYY-MM-DD'.
            three_days_prior (str): The date of the day 3 days prior in the format 'YYYY-MM-DD'.
        """
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def position_sizing(self):
        """
        Calculates the position size based on the current cash balance, risk factor, and last traded price.

        Returns:
            cash (float): The current cash balance.
            last_price (float): The last traded price of the asset.
            quantity (int): The quantity of the asset to buy.
        """
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity
    
    def get_sentiment(self):
        """
        Retrieves the sentiment analysis of news related to the symbol.

        Returns:
            probability (float): The probability of the sentiment analysis.
            sentiment (str): The sentiment of the news (positive, negative, neutral).
        """
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        """
        Executes a trading iteration based on the current market conditions.

        This method performs position sizing, retrieves sentiment analysis, and places buy/sell orders
        based on the calculated cash, sentiment, and probability.
        """
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > .999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"
        elif sentiment == "negative" and probability > .999:
            if self.last_trade == "buy":
                self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * .8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "buy"

start_date = datetime(2023,10,1)
end_date = datetime(2024,2,27)
broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":TICKER,
                                "cash_at_risk":.5})
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol":TICKER, "cash_at_risk":.5}
)

