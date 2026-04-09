import sys

class Tee:
    def __init__(self, filename, mode="a", buffering=1):
        # 使用行缓冲（buffering=1），确保每行立即写入
        self.file = open(filename, mode, encoding="utf-8", buffering=buffering)
        self.stdout = sys.stdout
        self.filename = filename

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        # 如果消息包含换行符，立即刷新以确保实时写入
        if '\n' in message:
            self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        """确保文件正确关闭"""
        if hasattr(self, 'file') and self.file:
            self.file.flush()
            self.file.close()
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        try:
            if hasattr(self, 'file') and self.file:
                self.file.flush()
                self.file.close()
        except:
            pass

if __name__ == "__main__":
    sys.stdout = Tee("run.log")
    sys.stderr = sys.stdout   # 可选，但强烈建议
# sys.stdout = Tee("run.log")
# sys.stderr = sys.stdout   # 可选，但强烈建议