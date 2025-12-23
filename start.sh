# 彻底清除所有常见的代理环境变量
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy

# 停止现有的streamlit进程
echo "正在停止现有的streamlit进程..."
pkill -9 -f "streamlit run furtures_terminal.py"
sleep 3

# 确认是否还有残留进程，如果有则强制终止
REMAINING=$(ps -ef | grep "streamlit run furtures_terminal.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$REMAINING" ]; then
    echo "发现残留进程，强制终止..."
    ps -ef | grep "streamlit run furtures_terminal.py" | grep -v grep | awk '{print $2}' | xargs kill -9
    sleep 2
fi
echo "所有streamlit进程已停止"

# 激活虚拟环境
source bin/activate

# 加载环境变量
if [ -f .env ]; then
    echo "正在加载数据库配置..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ 环境变量已加载"
    echo "  DB_HOST: ${DB_HOST}"
    echo "  DB_NAME: ${DB_NAME}"
    echo "  DB_USER: ${DB_USER}"
else
    echo "⚠️  警告: .env 文件不存在，数据库功能将不可用"
fi

# 启动streamlit服务
echo "正在启动streamlit服务..."
# 使用 env 命令确保环境变量传递给 nohup 进程
nohup env DB_HOST="$DB_HOST" DB_PORT="$DB_PORT" DB_NAME="$DB_NAME" DB_USER="$DB_USER" DB_PASSWORD="$DB_PASSWORD" SSL_MODE="$SSL_MODE" streamlit run furtures_terminal.py &
echo "服务已启动 (环境变量已传递)"
