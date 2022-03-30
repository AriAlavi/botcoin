from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
import asyncio

async def kline_listener(client):
    bm = BinanceSocketManager(client)
    symbol = 'BNBBTC'
    res_count = 0
    async with bm.kline_socket(symbol=symbol) as stream:
        while True:
            res = await stream.recv()
            res_count += 1
            print(res)
            if res_count == 5:
                res_count = 0
                order_book = await client.get_order_book(symbol=symbol)
                print(order_book)
async def main():
    # initialise the client
    client = await AsyncClient.create("", "") #api keys
    await kline_listener(client)
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
