import time

def get_time():
    return time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())

if __name__ == '__main__':
    print(get_time())
