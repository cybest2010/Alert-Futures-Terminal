# 🚀 A股期货情报终端 Pro

一个功能强大的股票期货分析工具，集成了实时快讯、形态选股、涨停分析、善庄狙击等多个实用模块。

## ✨ 核心功能

### 📰 实时快讯
- **股票快讯**: 实时抓取财联社、东方财富快讯，带NLP情感分析
- **期货快讯**: 上海金属网期货资讯，智能筛选相关内容
- **深度研报**: 微信公众号RSS订阅，精选投研文章

### 📊 数据分析
- **涨停天梯**: 实时涨停股票池，连板高度统计，封单强度分析
- **形态选股**: K线形态相似度匹配，基于Baostock历史数据
- **善庄狙击**: 追踪高胜率游资潜伏标的，多因子评分系统

### 🤖 交易辅助
- **模拟交易**: 股票/期货网格交易，ATR自适应网格间距
- **市场监控**: 实时市场指标监控（需数据库配置）

## 🚀 快速开始

### 环境要求
- Python 3.8+
- pip

### 安装依赖
```bash
cd streamlite-stock
pip install -r requirements.txt
```

### 配置（可选）
如需使用**善庄狙击**和**市场监控**功能，请配置数据库：

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入数据库信息
```

### 启动应用
```bash
streamlit run furtures_terminal.py
```

浏览器将自动打开 `http://localhost:8501`

## 📦 技术栈

- **前端框架**: Streamlit 1.28+
- **数据源**: AkShare, Baostock
- **数据库**: PostgreSQL + SQLAlchemy, SQLite
- **NLP**: Jieba分词
- **可视化**: Plotly
- **HTTP**: Requests + BeautifulSoup

## 📖 文档

- [快速启动指南](QUICK_START.md) - 详细的使用说明
- [善庄狙击文档](SMART_MONEY_SNIPER_README.md) - 核心功能详解

## ⚠️ 免责声明

本系统仅供学习研究使用，不构成任何投资建议。投资有风险，入市需谨慎。

## 📄 许可证

MIT License

## 🙏 致谢

Powered by Kilo Code AI Assistant | Version 5.0