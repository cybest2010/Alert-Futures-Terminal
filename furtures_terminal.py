import streamlit as st
import akshare as ak
import pandas as pd
import jieba
import jieba.analyse
import datetime
import time
import requests
from bs4 import BeautifulSoup
import re
import os
import baostock as bs
import sqlite3
import numpy as np
import concurrent.futures
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
import urllib3
import ssl
import logging
from sqlalchemy import create_engine, text

# 加载环境变量 (修复: 确保从.env文件加载数据库配置)
from dotenv import load_dotenv
load_dotenv()

# 忽略SSL证书验证警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 创建不验证SSL的会话
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. 配置与样式注入 (核心视觉优化)
# ==========================================
st.set_page_config(
    page_title="A股期货情报终端 Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入现代化金融终端 CSS
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* 隐藏默认的主菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 侧边栏美化 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    /* 顶部指标卡片化 */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        transition: all 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        border-color: #e5e7eb;
    }

    /* 新闻卡片优化 */
    .news-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        margin-bottom: 16px;
        border: 1px solid #f3f4f6;
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.08);
    }
    
    /* 侧边装饰条 */
    .card-border-indicator {
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .card-time {
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 500;
    }
    
    .card-badges {
        display: flex;
        gap: 6px;
    }

    .source-tag {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: #f3f4f6;
        color: #4b5563;
        font-weight: 600;
    }

    .signal-tag {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
    }

    .card-title {
        color: #111827;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 8px;
        line-height: 1.4;
    }

    .card-content {
        font-size: 0.9rem;
        color: #4b5563;
        line-height: 1.6;
        margin-bottom: 12px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .keyword-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    
    .keyword-tag {
        font-size: 0.75rem;
        color: #6b7280;
        background: #f9fafb;
        padding: 2px 8px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }

    /* 微信文章卡片优化 */
    .wechat-card {
        background: linear-gradient(145deg, #4f46e5, #7c3aed);
        color: white;
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.2);
        position: relative;
    }
    .wechat-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .wechat-meta {
        font-size: 0.8rem;
        opacity: 0.8;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .wechat-summary {
        font-size: 0.9rem;
        line-height: 1.5;
        opacity: 0.95;
        background: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .wechat-btn {
        display: inline-block;
        background: white;
        color: #7c3aed;
        padding: 6px 16px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 0.8rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .wechat-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    /* Tab 样式优化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        padding-bottom: 5px;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border: none;
        background-color: transparent;
        font-weight: 600;
        color: #6b7280;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心引擎层 (NLP + 数据库 + 选股算法)
# ==========================================

# 2.1 NLP 情感分析引擎
class SentimentEngine:
    def __init__(self):
        self.bullish_words = {
            "上涨", "突破", "利好", "支撑", "反弹", "增持", "买入", "做多", 
            "供不应求", "新高", "大涨", "回升", "走强", "红盘", "暴涨",
            "看多", "多头", "放量", "拉升", "飙升", "涨停", "强势", "连板"
        }
        self.bearish_words = {
            "下跌", "跌破", "利空", "压力", "回调", "减持", "卖出", "做空", 
            "库存积压", "跳水", "大跌", "回落", "走弱", "绿盘", "暴跌",
            "看空", "空头", "缩量", "杀跌", "跌停", "弱势", "崩盘"
        }
        jieba.initialize()

    def analyze(self, text):
        if not isinstance(text, str):
            return "观望", 0, []
        words = list(jieba.cut(text))
        score = 0
        keywords = []
        for word in words:
            if word in self.bullish_words:
                score += 1
                keywords.append(word)
            elif word in self.bearish_words:
                score -= 1
                keywords.append(word)
        if score > 0: signal = "做多"
        elif score < 0: signal = "做空"
        else: signal = "观望"
        return signal, score, list(set(keywords))

@st.cache_resource
def get_engine(): return SentimentEngine()
engine = get_engine()

# ==========================================
# 2.2.5 善庄狙击引擎 (Smart Money Sniper V5)
# ==========================================
class SmartMoneySniperV5:
    def __init__(self):
        # 数据库配置
        self.db_user = os.getenv("DB_USER", "postgres.xyafockjxvsfnuwlbslq")
        self.db_pass = os.getenv("DB_PASSWORD", "1qu23lis")
        self.db_host = os.getenv("DB_HOST", "aws-0-ap-southeast-1.pooler.supabase.com")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "stock_market")
        self.db_uri = f"postgresql+psycopg2://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
        
        self.engine = None
        self.broker_cache = {}

    def connect(self):
        try:
            self.engine = create_engine(self.db_uri, pool_pre_ping=True)
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False

    def get_broker_ranking_data(self):
        """统计所有活跃游资的 T+2 胜率"""
        if not self.engine:
            if not self.connect():
                return pd.DataFrame()
        
        sql = """
        SELECT
            dept_name,
            COUNT(*) as "操作次数",
            ROUND(SUM(CASE WHEN t2_pct > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 0) as "胜率"
        FROM lhb_detail
        WHERE trade_date >= CURRENT_DATE - INTERVAL '60 days'
          AND buy_amount > 0
          AND dept_name NOT LIKE '%%机构%%'
          AND dept_name NOT LIKE '%%股通%%'
        GROUP BY dept_name
        HAVING COUNT(*) >= 5
        ORDER BY "胜率" DESC;
        """
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
                cache = {}
                for _, row in df.iterrows():
                    cache[row['dept_name']] = int(row['胜率'])
                self.broker_cache = cache
                return df
        except Exception as e:
            print(f"获取胜率失败: {e}")
            return pd.DataFrame()

    def get_latent_stock_pool(self):
        """获取潜伏池"""
        self.get_broker_ranking_data()
        
        valid_brokers = [f"'{name}'" for name, win in self.broker_cache.items() if win >= 50]
        if not valid_brokers:
            return pd.DataFrame()
        
        broker_str = ",".join(valid_brokers)

        sql_stocks = f"""
        SELECT
            stock_code,
            stock_name,
            dept_name as "潜伏庄家",
            buy_amount as "买入金额"
        FROM lhb_detail
        WHERE trade_date >= CURRENT_DATE - INTERVAL '5 days'
          AND dept_name IN ({broker_str})
          AND buy_amount > 0
        ORDER BY trade_date DESC;
        """
        try:
            with self.engine.connect() as conn:
                df_pool = pd.read_sql(text(sql_stocks), conn)
            
            if not df_pool.empty:
                results = []
                for code, group in df_pool.groupby('stock_code'):
                    brokers = group['潜伏庄家'].unique()
                    max_win_rate = 0
                    broker_display_list = []
                    
                    for b in brokers:
                        win = self.broker_cache.get(b, 0)
                        if win > max_win_rate: max_win_rate = win
                        short_name = b.replace("证券股份有限公司", "").replace("有限责任公司", "").replace("营业部", "")[:6]
                        broker_display_list.append(f"{short_name}({win}%)")
                    
                    broker_display_list.sort(key=lambda x: int(x.split('(')[1][:-2]), reverse=True)
                    
                    results.append({
                        'stock_code': code,
                        'stock_name': group['stock_name'].iloc[0],
                        '潜伏庄家_fmt': " | ".join(broker_display_list[:2]),
                        '庄家最高胜率': max_win_rate,
                        '买入总额': group['买入金额'].sum()
                    })
                return pd.DataFrame(results)
            return pd.DataFrame()
        except Exception as e:
            print(f"获取潜伏池失败: {e}")
            return pd.DataFrame()

    def get_industry_info(self, stock_code):
        try:
            df = ak.stock_individual_info_em(symbol=stock_code)
            industry = df[df['item'] == '行业']['value'].values[0]
            return industry
        except:
            return "其他"

    def get_realtime_quotes(self, stock_codes):
        try:
            df_rt = ak.stock_zh_a_spot_em()
            target_df = df_rt[df_rt['代码'].isin(stock_codes)].copy()
            if target_df.empty: return pd.DataFrame()

            cols_map = {
                '代码': 'stock_code',
                '名称': 'stock_name_rt',
                '最新价': 'price',
                '今开': 'open_price',
                '昨收': 'prev_close',
                '成交额': 'amount',
                '量比': 'vol_ratio',
                '流通市值': 'mkt_cap_float'
            }
            target_df = target_df.rename(columns=cols_map)
            for col in ['amount', 'mkt_cap_float', 'vol_ratio', 'open_price']:
                if col in target_df.columns:
                    target_df[col] = pd.to_numeric(target_df[col], errors='coerce').fillna(0)
            return target_df
        except:
            return pd.DataFrame()

    def _get_plans(self, score, open_pct, amount_wan):
        buy_plan, exit_plan = "", ""
        
        if amount_wan > 10000 and open_pct > 3.0:
            buy_plan = "⚠️ [严禁追高] 巨量防砸！9:33站稳开盘价再看。"
        elif score >= 80 and 2 <= open_pct <= 6:
            buy_plan = "🔥 [激进] 9:30均线不破直接低吸。"
        elif score >= 60 and open_pct > 8:
            buy_plan = "⛔ [打板] 盈亏比差，仅做回封板。"
        elif score >= 50:
            buy_plan = "🛡️ [稳健] 黄线低吸，宁缺毋滥。"
        else:
            buy_plan = "👀 [观察] 破开盘价直接删。"

        if score >= 80:
            exit_plan = "T+1: 竞价若弱转强(高开)锁仓，否则止盈。"
        elif score >= 50:
            exit_plan = "T+1: 冲高3-5%止盈。破位止损。"
        else:
            exit_plan = "T+1: 不红盘直接竞价走。"
        
        return buy_plan, exit_plan

    def generate_signals(self):
        """生成交易信号"""
        df_pool = self.get_latent_stock_pool()
        if df_pool.empty:
            return pd.DataFrame()

        stock_codes = df_pool['stock_code'].tolist()
        df_rt = self.get_realtime_quotes(stock_codes)
        
        if df_rt.empty:
            return pd.DataFrame()

        df_final = pd.merge(df_pool, df_rt, on='stock_code', how='inner')
        results = []

        for idx, row in df_final.iterrows():
            score = 0
            
            open_price = row.get('open_price', 0)
            prev_close = row.get('prev_close', 0)
            amount = row.get('amount', 0)
            mkt_cap = row.get('mkt_cap_float', 0)
            vol_ratio = row.get('vol_ratio', 0)
            broker_max_win = row.get('庄家最高胜率', 0)
            
            if open_price == 0: continue

            open_pct = (open_price - prev_close) / prev_close * 100
            amount_wan = amount / 10000
            mkt_cap_yi = mkt_cap / 100000000
            auction_turnover = (amount / mkt_cap * 100) if mkt_cap > 0 else 0

            if amount_wan < 300: continue

            if amount_wan > 3000: score += 20
            elif amount_wan > 1000: score += 10
            
            if auction_turnover > 0.8: score += 20
            
            if vol_ratio > 5: score += 15
            elif vol_ratio > 2: score += 10
            
            if 1.0 <= open_pct <= 5.0: score += 25
            elif 5.0 < open_pct < 8.0:
                if amount_wan > 2000: score += 15
                else: score -= 10
            elif open_pct < -2: score -= 20
            
            if broker_max_win >= 80: score += 20
            elif broker_max_win >= 60: score += 10
            
            if mkt_cap_yi < 30: score += 5

            buy_plan, exit_plan = self._get_plans(score, open_pct, amount_wan)
            industry = self.get_industry_info(row['stock_code'])

            if score >= 40:
                results.append({
                    '代码': row['stock_code'],
                    '名称': row['stock_name'],
                    '潜伏庄家': row['潜伏庄家_fmt'],
                    '庄家最高胜率': broker_max_win,
                    '分数': score,
                    '流通市值(亿)': round(mkt_cap_yi, 1),
                    '行业': industry,
                    '涨幅%': round(open_pct, 2),
                    '竞价额': int(amount_wan),
                    '买入计划': buy_plan,
                    'T+1卖出': exit_plan
                })

        df_res = pd.DataFrame(results)
        if not df_res.empty:
            df_res = df_res.sort_values(
                by=['庄家最高胜率', '分数', '流通市值(亿)'],
                ascending=[False, False, True]
            )
        
        return df_res

# 2.2 本地数据库管理 (SQLite)
DB_NAME = "stocks.db"
TABLE_NAME = "daily_data"

class DataManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
            UNIQUE(symbol, date) ON CONFLICT REPLACE
        );
        """)
        self.cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_symbol ON {TABLE_NAME} (symbol);")
        self.conn.commit()

    def get_stock_data(self, symbol, lookback=None):
        if lookback:
            query = f"SELECT date, close FROM {TABLE_NAME} WHERE symbol=? ORDER BY date DESC LIMIT ?"
            df = pd.read_sql_query(query, self.conn, params=(symbol, lookback))
            if not df.empty:
                df = df.iloc[::-1].reset_index(drop=True)
        else:
            query = f"SELECT date, close FROM {TABLE_NAME} WHERE symbol=? ORDER BY date ASC"
            df = pd.read_sql_query(query, self.conn, params=(symbol,))
        return df

    def get_all_symbols(self):
        self.cursor.execute(f"SELECT DISTINCT symbol FROM {TABLE_NAME}")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_last_update_time(self):
        try:
            self.cursor.execute(f"SELECT MAX(date) FROM {TABLE_NAME}")
            return self.cursor.fetchone()[0]
        except:
            return "无数据"

# 2.3 形态选股：匹配算法
class PatternMatcher:
    @staticmethod
    def normalize(series):
        series = np.array(series)
        min_val = np.min(series)
        max_val = np.max(series)
        if max_val - min_val == 0: return np.zeros(len(series))
        return (series - min_val) / (max_val - min_val)

    @staticmethod
    def calculate_similarity(series_a, series_b):
        if len(series_a) != len(series_b) or len(series_a) < 3:
            return 0.0
        norm_a = PatternMatcher.normalize(series_a)
        norm_b = PatternMatcher.normalize(series_b)
        try:
            # 1 - 余弦距离 = 相似度
            sim = 1 - cosine(norm_a, norm_b)
            return max(0, sim * 100)
        except:
            return 0.0

# 2.4 Baostock 下载 Worker
def download_worker(symbol, start_date, end_date):
    """
    使用 Baostock 下载数据 (线程安全模式，每次调用独立登录)
    symbol 格式: "sh.600000" 或 "sz.000001"
    """
    lg = bs.login()
    if lg.error_code != '0': return None

    try:
        # Baostock 日期格式 YYYY-MM-DD
        s_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        e_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"

        # 确保 symbol 有前缀
        clean_symbol = symbol.replace("sh", "").replace("sz", "").replace(".", "")
        if symbol.startswith("sh") or symbol.startswith("6"):
            bs_symbol = f"sh.{clean_symbol}"
        else:
            bs_symbol = f"sz.{clean_symbol}"

        # 前复权下载
        rs = bs.query_history_k_data_plus(
            bs_symbol, "date,open,high,low,close,volume",
            start_date=s_date, end_date=e_date, frequency="d", adjustflag="2"
        )

        if rs.error_code != '0': 
            bs.logout()
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        bs.logout()

        if not data_list: return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 转换类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 格式化
        df['date'] = df['date'].str.replace("-", "")
        df['symbol'] = bs_symbol # 存入完整代码
        return df

    except Exception:
        bs.logout()
        return None

# ==========================================
# 3. 数据处理与获取 (快讯/RSS/涨停)
# ==========================================
def standardize_dataframe(df, source_name, category_name):
    datetime_candidates = ['时间', '发布时间', '更新时间', 'datetime', 'time', '日期', 'updated']
    content_candidates = ['内容', '新闻内容', '标题', '新闻标题', 'content', 'title', '正文', 'description']
    
    datetime_col = None
    content_col = None
    
    for col in datetime_candidates:
        if col in df.columns:
            datetime_col = col
            break
    
    for col in content_candidates:
        if col in df.columns:
            content_col = col
            break
    
    if datetime_col is None or content_col is None:
        return pd.DataFrame() 
    
    df = df.rename(columns={datetime_col: 'datetime', content_col: 'content'})
    if 'datetime' in df.columns: df['datetime'] = df['datetime'].astype(str)
    
    df['source'] = source_name
    df['category'] = category_name
    return df[['datetime', 'content', 'source', 'category']]

@st.cache_data(ttl=600)
def fetch_wechat_rss():
    """微信RSS文章获取 - 增强容错性"""
    try:
        # 主数据源
        articles = get_wechat_rss_primary()
        if articles:
            return articles
    except Exception as e:
        print(f"主RSS源失败: {e}")
    
    try:
        # 备用数据源：多个RSS地址
        articles = get_wechat_rss_backup()
        if articles:
            return articles
    except Exception as e:
        print(f"备用RSS源失败: {e}")
    
    # 返回模拟数据，确保界面不空白
    return get_mock_wechat_articles()

def get_wechat_rss_primary():
    """主RSS数据源"""
    url = 'https://wewerss.168689.xyz/feed/all.json'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://wewerss.168689.xyz/'
    }
    
    response = requests.get(url, headers=headers, timeout=8)
    if response.status_code != 200:
        return []
    
    items = response.json().get('items', [])
    articles = []
    
    for item in items[:30]:
        try:
            content = item.get('content', '')
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                content_text = soup.get_text(separator=' ', strip=True)
            else:
                content_text = ''
                
            articles.append({
                'title': item.get('title', '无标题'),
                'channel_name': item.get('channel_name', '精选文章'),
                'updated': item.get('updated', '')[:16].replace('T', ' ') if item.get('updated') else '',
                'link': item.get('link', ''),
                'summary': (content_text[:300] + "...") if len(content_text) > 300 else content_text
            })
        except Exception as e:
            print(f"处理RSS项目时出错: {e}")
            continue
    
    return articles

def get_wechat_rss_backup():
    """备用RSS数据源"""
    backup_urls = [
        'https://rsshub.app/weixin/sogou/zhifujing',
        'https://rsshub.app/weixin/sogou/lcj147258369'
    ]
    
    for url in backup_urls:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                # 解析RSS XML
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                articles = []
                for item in items[:20]:
                    title = item.find('title')
                    link = item.find('link')
                    pubDate = item.find('pubDate')
                    description = item.find('description')
                    
                    if title and link:
                        summary = description.get_text() if description else ''
                        articles.append({
                            'title': title.get_text(),
                            'channel_name': '财经精选',
                            'updated': pubDate.get_text()[:16] if pubDate else '',
                            'link': link.get_text(),
                            'summary': summary[:300] + "..." if len(summary) > 300 else summary
                        })
                
                if articles:
                    return articles
        except Exception as e:
            print(f"备用RSS源 {url} 失败: {e}")
            continue
    
    return []

def get_mock_wechat_articles():
    """当所有数据源都失败时，返回模拟数据"""
    import datetime
    now = datetime.datetime.now()
    
    mock_articles = [
        {
            'title': '市场热点：新能源板块迎来新机遇',
            'channel_name': '财经观察',
            'updated': now.strftime('%Y-%m-%d %H:%M'),
            'link': '#',
            'summary': '随着政策支持力度加大，新能源汽车产业链各环节投资机会凸显，电池技术、光伏发电等领域前景看好...'
        },
        {
            'title': 'A股收评：三大指数涨跌不一，结构性机会显现',
            'channel_name': '市场分析',
            'updated': now.strftime('%Y-%m-%d %H:%M'),
            'link': '#',
            'summary': '今日A股市场呈现震荡整理态势，题材股活跃度提升，建议关注业绩确定性较强的优质标的...'
        },
        {
            'title': '期货市场：金属期货价格波动分析',
            'channel_name': '期货研究',
            'updated': now.strftime('%Y-%m-%d %H:%M'),
            'link': '#',
            'summary': '受宏观经济因素影响，主要金属期货品种价格波动加剧，投资者需注意风险控制...'
        }
    ]
    return mock_articles

@st.cache_data(ttl=15)
def fetch_limitup_data():
    """涨停数据获取 - 增强容错性"""
    try:
        # 主数据源
        df = get_limitup_from_xuangubao()
        if not df.empty:
            return df
    except Exception as e:
        print(f"主数据源失败: {e}")
    
    try:
        # 备用数据源：使用akshare
        df = get_limitup_from_akshare()
        if not df.empty:
            return df
    except Exception as e:
        print(f"备用数据源失败: {e}")
    
    return pd.DataFrame()

def get_limitup_from_xuangubao():
    """从选股宝API获取涨停数据"""
    url = "https://flash-api.xuangubao.cn/api/pool/detail?pool_name=limit_up"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://xuangubao.cn/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=8, verify=False)
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json().get('data', [])
        if not data:
            return pd.DataFrame()
    except Exception as e:
        print(f"选股宝API请求失败: {e}")
        return pd.DataFrame()
    
    records = []
    for stock in data:
        try:
            last_limit_up_ts = stock.get('last_limit_up', 0)
            last_limit_time = datetime.datetime.fromtimestamp(last_limit_up_ts).strftime('%H:%M:%S') if last_limit_up_ts else '--'
            records.append({
                '股票名称': stock.get('stock_chi_name', '未知'),
                '代码': str(stock.get('symbol', ''))[:6],
                '当前价': round(float(stock.get('price', 0)), 2),
                '流通市值(亿)': round(float(stock.get('non_restricted_capital', 0)) / 100000000, 2),
                '连续涨停天数': int(stock.get('limit_up_days', 0)),
                '涨停原因': stock.get('surge_reason', {}).get('stock_reason', '无') if isinstance(stock.get('surge_reason'), dict) else str(stock.get('surge_reason', '无')),
                '换手率': float(stock.get('turnover_ratio', 0)),
                '买盘封单比': round(float(stock.get('buy_lock_volume_ratio', 0)), 2),
                '最后涨停时间': last_limit_time,
            })
        except Exception as e:
            print(f"处理股票数据时出错: {e}")
            continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df = df.sort_values(by='连续涨停天数', ascending=False).head(10).reset_index(drop=True)
    return df

def get_limitup_from_akshare():
    """从akshare获取涨停数据作为备用"""
    try:
        # 尝试获取实时涨停股票数据
        df = ak.stock_zt_pool_strong_em()
        if df.empty:
            return pd.DataFrame()
        
        # 标准化列名
        column_mapping = {
            '代码': '代码',
            '名称': '股票名称',
            '现价': '当前价',
            '涨幅': '涨幅',
            '换手': '换手率',
            '概念': '涨停原因'
        }
        
        # 选择需要的列
        available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_selected = df[list(available_cols.keys())].rename(columns=available_cols)
        
        # 添加计算列
        df_selected['连续涨停天数'] = 1  # akshare数据中涨停天数信息有限，默认为1
        df_selected['流通市值(亿)'] = 0  # akshare可能没有市值数据
        df_selected['买盘封单比'] = 0
        
        return df_selected.head(10).reset_index(drop=True)
    except Exception as e:
        print(f"akshare涨停数据获取失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_sh_sentiment():
    """获取全市场下跌占比 - 多数据源容错"""
    # 方法1: 尝试东方财富实时数据
    try:
        print("[DEBUG] 正在获取全市场下跌占比数据...")
        df = ak.stock_zh_a_spot_em()
        
        if not df.empty and '涨跌幅' in df.columns:
            total = len(df)
            decline = len(df[df['涨跌幅'] < 0])
            ratio = (decline / total) * 100 if total > 0 else 50.0
            print(f"[DEBUG] ✓ 成功获取: 总股票={total}, 下跌={decline}, 占比={ratio:.1f}%")
            return ratio, 0.0
    except Exception as e:
        print(f"[DEBUG] 东方财富API失败: {type(e).__name__}")
    
    # 方法2: 尝试使用新浪接口
    try:
        import requests
        url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=40&sort=symbol&asc=1&node=hs_a&symbol=&_s_r_a=init"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            import json
            data = json.loads(response.text)
            if data:
                decline_count = sum(1 for item in data if float(item.get('changepercent', 0)) < 0)
                ratio = (decline_count / len(data)) * 100
                print(f"[DEBUG] ✓ 新浪接口成功: 下跌占比={ratio:.1f}%")
                return ratio, 0.0
    except Exception as e:
        print(f"[DEBUG] 新浪接口失败: {type(e).__name__}")
    
    # 方法3: 返回默认值并显示警告
    print("[WARNING] 所有行情API均不可用，返回默认值50%")
    return 50.0, 0.0

@st.cache_data(ttl=300)
def fetch_market_monitoring_from_db():
    try:
        import psycopg2
        # 诊断日志：检查环境变量
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        
        print(f"[DEBUG] 数据库连接参数检查:")
        print(f"  DB_HOST: {db_host if db_host else '❌ 未设置'}")
        print(f"  DB_NAME: {db_name if db_name else '❌ 未设置'}")
        print(f"  DB_USER: {db_user if db_user else '❌ 未设置'}")
        
        if not all([db_host, db_name, db_user]):
            print("[ERROR] 数据库环境变量缺失，无法连接PostgreSQL")
            return None
        
        conn = psycopg2.connect(
            host=db_host,
            port=os.getenv("DB_PORT", 5432),
            dbname=db_name,
            user=db_user,
            password=os.getenv("DB_PASSWORD"),
            sslmode=os.getenv("SSL_MODE", "require")
        )
        print("[DEBUG] ✓ PostgreSQL连接成功")
        
        query = "SELECT record_date, micro_volatility, micro_monthly_return, strong_industries, high_div_ratio, low_turn_ratio, jpbd_511010, jpbd_shanghai, signals FROM market_monitoring ORDER BY record_date DESC LIMIT 1"
        with conn.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
            if row:
                print(f"[DEBUG] ✓ 查询到数据: 日期={row[0]}, 微盘波动率={row[1]}, JPBD={row[7]}")
                return {
                    'record_date': row[0], 'micro_volatility': row[1], 'micro_monthly_return': row[2],
                    'strong_industries': row[3] or [], 'high_div_ratio': row[4], 'low_turn_ratio': row[5],
                    'jpbd_511010': row[6], 'jpbd_shanghai': row[7], 'signals': row[8] or []
                }
            else:
                print("[ERROR] ❌ market_monitoring表为空，无数据记录")
        conn.close()
    except Exception as e:
        print(f"[ERROR] 数据库访问失败: {type(e).__name__}: {str(e)}")
    return None

@st.cache_data(ttl=300)
def fetch_stock_news():
    all_news = []
    
    # 使用线程池并行获取数据，提高响应速度
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        
        # 提交财联社快讯任务
        future1 = executor.submit(get_stock_news_cls_safe)
        futures.append(('财联社', future1))
        
        # 提交东方财富快讯任务
        future2 = executor.submit(get_stock_news_em_safe)
        futures.append(('东方财富', future2))
        
        # 提交备用数据源任务
        future3 = executor.submit(get_stock_news_backup)
        futures.append(('备用源', future3))
        
        # 收集结果
        for source_name, future in futures:
            try:
                df = future.result(timeout=8)  # 设置8秒超时
                if not df.empty:
                    df['source'] = source_name
                    df['category'] = '股票'
                    all_news.append(df)
            except Exception as e:
                print(f"数据源 {source_name} 获取失败: {e}")
                continue

    if all_news:
        combined = pd.concat(all_news, ignore_index=True).drop_duplicates(subset=['content'])
        try: combined = combined.sort_values('datetime', ascending=False)
        except: pass
        return combined.head(40)
    return pd.DataFrame()

def get_stock_news_cls_safe():
    """安全获取财联社快讯"""
    try:
        # 设置超时和SSL验证参数
        df = ak.stock_info_global_cls()
        if df.empty:
            return pd.DataFrame()
        return standardize_dataframe(df, '财联社', '股票')
    except Exception as e:
        print(f"财联社快讯获取失败: {e}")
        return pd.DataFrame()

def get_stock_news_em_safe():
    """安全获取东方财富快讯"""
    try:
        # 设置超时和SSL验证参数
        df = ak.stock_info_global_em()
        if df.empty:
            return pd.DataFrame()
        return standardize_dataframe(df, '东方财富', '股票')
    except Exception as e:
        print(f"东方财富快讯获取失败: {e}")
        return pd.DataFrame()

def get_stock_news_backup():
    """备用快讯数据源"""
    try:
        # 尝试使用新浪财经API作为备用
        import requests
        url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=1686&k=&num=20&page=1"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/"
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            records = []
            for item in data.get('result', {}).get('data', []):
                records.append({
                    'datetime': item.get('ctime', ''),
                    'content': item.get('title', '') + ' - ' + item.get('intro', ''),
                    'source': '新浪财经',
                    'category': '股票'
                })
            return pd.DataFrame(records)
    except Exception as e:
        print(f"备用数据源失败: {e}")
        return pd.DataFrame()
@st.cache_data(ttl=300)
def fetch_futures_news():
    all_news = []
    
    # 使用线程池并行获取期货数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        # 提交上海金属网期货快讯任务
        future1 = executor.submit(get_futures_news_shmet_safe)
        futures.append(('上海金属网', future1))
        
        # 提交财联社期货快讯任务
        future2 = executor.submit(get_futures_news_cls_safe)
        futures.append(('财联社', future2))
        
        # 收集结果
        for source_name, future in futures:
            try:
                df = future.result(timeout=8)
                if not df.empty:
                    df['source'] = source_name
                    df['category'] = '期货'
                    all_news.append(df)
            except Exception as e:
                print(f"期货数据源 {source_name} 获取失败: {e}")
                continue

    if all_news:
        combined = pd.concat(all_news, ignore_index=True).drop_duplicates(subset=['content'])
        try: combined = combined.sort_values('datetime', ascending=False)
        except: pass
        return combined.head(30)
    return pd.DataFrame()

def get_futures_news_shmet_safe():
    """安全获取上海金属网期货快讯"""
    try:
        # 设置超时和SSL验证参数
        df = ak.futures_news_shmet()
        if df.empty:
            return pd.DataFrame()
        return standardize_dataframe(df, '上海金属网', '期货')
    except Exception as e:
        print(f"上海金属网期货快讯获取失败: {e}")
        return pd.DataFrame()

def get_futures_news_cls_safe():
    """从财联社快讯中筛选期货相关内容"""
    try:
        # 设置超时和SSL验证参数
        df = ak.stock_info_global_cls()
        if df.empty:
            return pd.DataFrame()
        
        # 使用更全面的期货关键词
        futures_kw = ['期货', 'IF', 'IC', 'IM', 'IH', 'CU', 'AL', 'ZN', 'PB', 'AU', 'AG',
                     'RU', 'FU', 'BU', 'SC', 'TA', 'MA', 'CF', 'SR', 'AP', 'CJ', 'RM',
                     'OI', 'RS', 'RM', 'MA', 'TA', 'PF', 'EG', 'EB', 'EC', 'PF', 'LU',
                     '铜', '铝', '锌', '铅', '黄金', '白银', '原油', '螺纹', '铁矿',
                     '焦炭', '焦煤', '玻璃', '甲醇', 'pta', '白糖', '棉花', '苹果',
                     '红枣', '菜粕', '玉米', '豆粕', '豆油', '棕榈', '乙烯', '丙烯']
        
        # 先标准化数据框架
        df_std = standardize_dataframe(df, '财联社', '期货')
        if df_std.empty:
            return pd.DataFrame()
        
        # 筛选包含期货关键词的内容
        mask = df_std['content'].str.contains('|'.join(futures_kw), na=False, case=False)
        df_filtered = df_std[mask]
        
        return df_filtered.head(20)
    except Exception as e:
        print(f"财联社期货快讯筛选失败: {e}")
        return pd.DataFrame()


# ==========================================
# 4. 前端渲染逻辑
# ==========================================
def render_card(row):
    content = row['content']
    time_str = str(row['datetime'])[-8:]
    source = row['source']
    signal, score, keywords = engine.analyze(content)
    
    if signal == "做多":
        indicator_color = "#ef4444" 
        bg_badge = "#fef2f2"
        text_badge = "#dc2626"
        score_txt = f"+{score}"
    elif signal == "做空":
        indicator_color = "#10b981"
        bg_badge = "#ecfdf5"
        text_badge = "#059669"
        score_txt = f"{score}"
    else:
        indicator_color = "#9ca3af"
        bg_badge = "#f3f4f6"
        text_badge = "#4b5563"
        score_txt = "0"

    keywords_html = ""
    for k in keywords[:4]:
        keywords_html += f'<span class="keyword-tag">{k}</span>'

    html = f"""
    <div class="news-card">
        <div class="card-border-indicator" style="background-color: {indicator_color};"></div>
        <div class="card-header">
            <div class="card-badges">
                <span class="source-tag">{source}</span>
                <span class="signal-tag" style="background-color: {bg_badge}; color: {text_badge};">
                    {signal} {score_txt}
                </span>
            </div>
            <span class="card-time">{time_str}</span>
        </div>
        <div class="card-title">{content[:40]}{'...' if len(content)>40 else ''}</div>
        <div class="card-content">{content}</div>
        <div class="keyword-container">
            {keywords_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_wechat_article(article):
    html = f"""
    <div class="wechat-card">
        <div class="wechat-title">📱 {article['title']}</div>
        <div class="wechat-meta">
            <span>👤 {article['channel_name']}</span>
            <span>🕒 {article['updated']}</span>
        </div>
        <div class="wechat-summary">
            {article['summary']}
        </div>
        <a href="{article['link']}" target="_blank" class="wechat-btn">
            阅读原文 ➔
        </a>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 6. 模拟交易引擎 (期货增强版)
# ==========================================
# --- A. 稳健的实时行情获取模块 (基于你提供的代码优化) ---

def get_futures_price_robust(symbol):
    """
    获取期货实时价格 (多源容错版)
    输入: RB2505
    输出: float 价格
    """
    # 1. 优先尝试新浪 HTTP 接口 (速度最快)
    try:
        # 新浪接口要求大写，如 nf_RB2505
        url = f"http://hq.sinajs.cn/list=nf_{symbol.upper()}"
        headers = {"Referer": "http://finance.sina.com.cn/"}
        resp = requests.get(url, headers=headers, timeout=2)
        
        if resp.status_code == 200 and '="' in resp.text:
            data_str = resp.text.split('="')[1].strip('"')
            data = data_str.split(',')
            
            # 新浪期货字段解析:
            # [0]名称 [6]买价 [7]卖价 [8]最新价 [14]成交量 ...
            # 优先取最新价(8)，如果是0则取买一(6)或卖一(7)
            price = 0.0
            if len(data) > 8:
                try:
                    price = float(data[8]) # 最新价
                except: pass
                
                if price == 0 and len(data) > 6:
                    try: price = float(data[6]) # 买一
                    except: pass
                
                if price == 0 and len(data) > 7:
                    try: price = float(data[7]) # 卖一
                    except: pass
                    
            if price > 0:
                return price
    except Exception:
        pass

    # 2. 失败则尝试 AkShare (备用)
    try:
        # 尝试获取主力合约 (如果是 RB0 这种格式)
        if symbol.endswith('0'):
            df = ak.futures_main_sina(symbol=symbol)
            if not df.empty:
                return float(df['close'].iloc[-1])
        else:
            # 尝试具体合约
            # 注意：akshare 具体合约接口较慢，作为最后的兜底
            pass 
    except:
        pass
        
    return None

def get_stock_price_realtime(code):
    """股票实时价格 (保持原有的东财接口)"""
    try:
        df = ak.stock_zh_a_spot_em()
        row = df[df['代码'] == code]
        if not row.empty:
            return float(row['最新价'].values[0])
    except:
        pass
    return None

def calculate_indicators_pro(code, asset_type='股票'):
    """
    计算 ATR 和 ADX (智能匹配历史数据)
    特点：如果是期货具体合约(RB2505)，会自动映射到主力连续(RB0)来计算指标
    """
    try:
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=200)).strftime("%Y%m%d")
        df = pd.DataFrame()

        if asset_type == '股票':
            prefix = "sh" if code.startswith("6") else "sz"
            full_code = f"{prefix}{code}" if not code.startswith(("sh", "sz")) else code
            df = ak.stock_zh_a_hist_tx(symbol=full_code, start_date=start_date, end_date=end_date, adjust="qfq")
            if not df.empty:
                df.rename(columns={'close':'收盘','high':'最高','low':'最低','open':'开盘'}, inplace=True)

        elif asset_type == '期货':
            # 智能映射：RB2505 -> RB0 (主力连续)
            # 正则提取字母部分
            match = re.match(r"([A-Za-z]+)", code)
            if match:
                product_code = match.group(1).upper() # RB
                main_code = f"{product_code}0" # RB0
            else:
                main_code = code # 兜底

            # 获取期货日线
            df = ak.futures_main_sina(symbol=main_code)
            if not df.empty:
                df = df.tail(100).copy()

        if df.empty or len(df) < 30: return None, None, None, None

        # --- 通用指标计算 ---
        df['H-L'] = df['最高'] - df['最低']
        df['H-PC'] = abs(df['最高'] - df['收盘'].shift(1))
        df['L-PC'] = abs(df['最低'] - df['收盘'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        current_atr = df['TR'].tail(14).mean()
        
        # ADX
        df['Up'] = df['最高'] - df['最高'].shift(1)
        df['Down'] = df['最低'].shift(1) - df['最低']
        df['+DM'] = np.where((df['Up']>df['Down']) & (df['Up']>0), df['Up'], 0.0)
        df['-DM'] = np.where((df['Down']>df['Up']) & (df['Down']>0), df['Down'], 0.0)
        tr14 = df['TR'].tail(14).sum()
        pdm14 = df['+DM'].tail(14).sum()
        mdm14 = df['-DM'].tail(14).sum()
        
        if tr14 == 0 or (pdm14 + mdm14) == 0: return current_atr, 0, 0, 0

        pdi = 100 * pdm14 / tr14
        mdi = 100 * mdm14 / tr14
        dx = 100 * abs(pdi - mdi) / (pdi + mdi)
        current_adx = dx 
        
        return current_atr, current_adx, pdi, mdi
    except Exception as e:
        return None, None, None, None

# --- B. 交易引擎类 ---

class GridTraderEngine:
    def __init__(self, config, log_container, status_container):
        self.cfg = config
        self.cash = config['initial_cash']
        self.holdings = 0 
        self.base_price = None
        self.grid_gap = 0.0
        self.can_buy = True
        self.can_sell = True
        self.logs = []
        
        self.asset_type = config.get('asset_type', '股票')
        self.multiplier = config.get('multiplier', 1)
        self.margin_rate = config.get('margin_rate', 1.0)
        
        self.log_container = log_container
        self.status_container = status_container
        
        self.add_log(f"🤖 引擎启动 | 目标: {config['symbol']} ({self.asset_type})")
        
        # 初始化指标
        with st.spinner(f"正在计算 {config['symbol']} 的 ATR/ADX 指标..."):
            success = self.update_indicators()
            if not success:
                self.add_log("⚠️ 警告: 历史数据不足，使用默认网格参数")
                self.grid_gap = config.get('default_price', 100) * 0.01 # 兜底

    def add_log(self, msg):
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.insert(0, f"[{time_str}] {msg}")
        with self.log_container.container():
            for log in self.logs[:50]:
                if "买入" in log or "开多" in log:
                    st.success(log)
                elif "卖出" in log or "平多" in log:
                    st.error(log)
                elif "风控" in log:
                    st.warning(log)
                else:
                    st.info(log)

    def update_indicators(self):
        atr, adx, pdi, mdi = calculate_indicators_pro(self.cfg['symbol'], self.asset_type)
        if atr:
            self.grid_gap = atr * 1.0 # 1倍ATR作为间距
            is_strong = adx > self.cfg['adx_threshold']
            self.can_buy = not (is_strong and mdi > pdi)
            self.can_sell = not (is_strong and pdi > mdi)
            return True
        return False

    def trade(self, direction, price):
        if price <= 0: return

        # 仓位计算
        if self.asset_type == '股票':
            vol = int(self.cfg['grid_amt'] / price / 100) * 100
            margin_needed = vol * price
        else:
            # 期货: 金额 / (价格 * 乘数 * 保证金率)
            one_hand_cost = price * self.multiplier * self.margin_rate
            vol = int(self.cfg['grid_amt'] / one_hand_cost)
            margin_needed = vol * one_hand_cost
            
        if vol == 0: return

        if direction == 'BUY':
            if self.cash >= margin_needed:
                self.cash -= margin_needed
                self.holdings += vol
                self.base_price -= self.grid_gap
                act = "买入" if self.asset_type=='股票' else "开多"
                self.add_log(f"⚡ {act} | 价:{price} | 量:{vol} | 额:{margin_needed:.0f}")
            else:
                self.add_log("⚠️ 资金不足")
                
        elif direction == 'SELL':
            if self.holdings >= vol:
                if self.asset_type == '股票':
                    self.cash += vol * price
                else:
                    # 期货平仓: 释放保证金 + 盈亏
                    # 盈亏 = (卖价 - 买入基准) * 乘数 * 手数
                    # 释放保证金 = vol * price * multiplier * margin (近似)
                    released_margin = vol * price * self.multiplier * self.margin_rate
                    # 这里的基准价近似为成本价
                    profit = (price - (price - self.grid_gap)) * self.multiplier * vol
                    self.cash += (released_margin + profit)

                self.holdings -= vol
                self.base_price += self.grid_gap
                act = "卖出" if self.asset_type=='股票' else "平多"
                self.add_log(f"💰 {act} | 价:{price} | 量:{vol}")

    def run_step(self, current_price):
        # 1. 资产状态计算
        if self.asset_type == '股票':
            total_asset = self.cash + (self.holdings * current_price)
        else:
            # 期货权益 = 现金 + 保证金 + 浮动盈亏
            float_pnl = 0
            if self.base_price:
                float_pnl = (current_price - self.base_price) * self.holdings * self.multiplier
            
            used_margin = self.holdings * current_price * self.multiplier * self.margin_rate
            # 简化显示：总权益
            total_asset = self.cash + used_margin + float_pnl

        ret = (total_asset - self.cfg['initial_cash']) / self.cfg['initial_cash'] * 100
        
        # 2. UI 更新
        with self.status_container.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最新价", f"{current_price}", f"基准: {self.base_price:.2f}" if self.base_price else "--")
            
            unit = "股" if self.asset_type=='股票' else "手"
            val_label = "市值" if self.asset_type=='股票' else "占用保证金"
            val_num = self.holdings * current_price * self.multiplier * self.margin_rate
            
            c2.metric(f"持仓({unit})", f"{self.holdings}", f"{val_label}: {val_num:.0f}")
            c3.metric("可用资金", f"{self.cash:.0f}")
            c4.metric("总权益", f"{total_asset:.0f}", f"{ret:.2f}%")
            
            st.caption(f"网格间距: {self.grid_gap:.2f} | 状态: {'运行中' if self.can_buy else '风控拦截'}")

        # 3. 首次定基准
        if self.base_price is None:
            self.base_price = current_price
            self.add_log(f"🏁 初始基准价锁定: {current_price}")
            return

        # 4. 交易触发
        if current_price <= self.base_price - self.grid_gap:
            if self.can_buy: self.trade('BUY', current_price)
        elif current_price >= self.base_price + self.grid_gap:
            if self.can_sell: self.trade('SELL', current_price)
            
# ==========================================
# 5. 主界面构建
# ==========================================
def main():
    # 初始化形态选股工具
    db_manager = DataManager()
    matcher = PatternMatcher()

    # --- 侧边栏优化 ---
    with st.sidebar:
        st.header("🎛️ 控制台")
        
        st.subheader("数据源")
        st.markdown("""
        <div style="font-size:0.9rem; color:#6b7280; margin-bottom:10px;">
        • <b>股票</b>：财联社、东方财富<br>
        • <b>期货</b>：上海金属网、NLP筛选<br>
        • <b>深度</b>：精选微信投研文章<br>
        • <b>选股</b>：Baostock + 本地数据库
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("市场监控")
        st.info("🟢 实时监控运行中")

        st.divider()
        filter_option = st.radio("🔍 信号过滤", ["全部", "只看做多 (Bull)", "只看做空 (Bear)"], index=0)
        
        st.divider()
        if st.button("🔄 刷新全站数据", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            
        st.caption(f"Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}")

    # --- 顶部仪表盘 ---
    st.title("A股期货情报终端 Pro")
    
    with st.spinner("正在聚合市场数据..."):
        market_data = fetch_market_monitoring_from_db()
        sh_decline_ratio, _ = fetch_sh_sentiment()
        
        # 诊断日志：检查返回的数据
        print(f"[DEBUG] 市场数据获取结果: market_data={'有数据' if market_data else '❌ None'}")
        if market_data:
            print(f"  - micro_volatility: {market_data.get('micro_volatility')}")
            print(f"  - jpbd_shanghai: {market_data.get('jpbd_shanghai')}")
            print(f"  - low_turn_ratio: {market_data.get('low_turn_ratio')}")
        print(f"  - sh_decline_ratio: {sh_decline_ratio:.1f}%")

    # 显示数据源状态提示
    if not market_data:
        st.warning("⚠️ PostgreSQL数据库未配置，部分指标显示默认值。请设置环境变量: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD")

    # First row: 4 main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = market_data['micro_volatility']*100 if (market_data and market_data.get('micro_volatility')) else 0
        delta = market_data['micro_monthly_return']*100 if (market_data and market_data.get('micro_monthly_return')) else 0
        st.metric("微盘股波动率", f"{val:.2f}%", f"{delta:+.2f}% 月度", delta_color="inverse")
    with col2:
        val = market_data['jpbd_shanghai'] if (market_data and market_data.get('jpbd_shanghai')) else 0
        state = "超买" if val > 80 else "正常"
        st.metric("上证 JPBD", f"{val:.1f}", state, delta_color="inverse" if val > 80 else "normal")
    with col3:
        st.metric("全市场下跌占比", f"{sh_decline_ratio:.1f}%", "实时风控", delta_color="inverse")
    with col4:
        val = market_data['low_turn_ratio']*100 if (market_data and market_data.get('low_turn_ratio')) else 0
        st.metric("低换手股比例", f"{val:.1f}%", "流动性监测")

    # Second row: Additional metrics (previously missing)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        industries = market_data.get('strong_industries', []) if market_data else []
        industry_text = industries[0] if industries else "无数据"
        st.metric("强势传统行业", industry_text, f"共{len(industries)}个")
    with col6:
        val = market_data['high_div_ratio']*100 if (market_data and market_data.get('high_div_ratio')) else 0
        st.metric("高股息蓝筹占比", f"{val:.2f}%", "防御指标")
    with col7:
        val = market_data['jpbd_511010'] if (market_data and market_data.get('jpbd_511010')) else 0
        state = "高位" if val > 60 else "正常"
        st.metric("国债JPBD值", f"{val:.2f}", state)
    with col8:
        signals = market_data.get('signals', []) if market_data else []
        signal_status = "🚨 有警报" if signals else "✅ 无警报"
        signal_count = len(signals) if signals else 0
        st.metric("警报信号", signal_status, f"{signal_count}条")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Display detailed alerts if any
    if market_data and market_data.get('signals'):
        for signal in market_data['signals']:
            st.warning(f"🚨 **系统警报**: {signal}")

    # --- 主要内容区 (新增形态选股 Tab + 善庄狙击) ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["⚡ 股票快讯", "🏗️ 期货快讯", "🧠 深度研报", "🔥 涨停天梯", "📈 形态选股", "🎯 善庄狙击", "🤖 模拟交易"])
    # === Tab 1: 股票快讯 ===
    with tab1:
        df = fetch_stock_news()
        if df.empty:
            st.info("📭 暂无最新快讯")
        else:
            c1, c2 = st.columns(2)
            for idx, row in df.iterrows():
                signal, _, _ = engine.analyze(row['content'])
                if "做多" in filter_option and signal != "做多": continue
                if "做空" in filter_option and signal != "做空": continue
                with c1 if idx % 2 == 0 else c2:
                    render_card(row)

    # === Tab 2: 期货快讯 ===
    with tab2:
        df = fetch_futures_news()
        if df.empty:
            st.info("📭 暂无期货快讯")
        else:
            c1, c2 = st.columns(2)
            for idx, row in df.iterrows():
                signal, _, _ = engine.analyze(row['content'])
                if "做多" in filter_option and signal != "做多": continue
                if "做空" in filter_option and signal != "做空": continue
                with c1 if idx % 2 == 0 else c2:
                    render_card(row)

    # === Tab 3: 微信研报 ===
    with tab3:
        articles = fetch_wechat_rss()
        if not articles:
            st.info("📭 暂无文章更新")
        else:
            c1, c2 = st.columns(2)
            for idx, article in enumerate(articles):
                with c1 if idx % 2 == 0 else c2:
                    render_wechat_article(article)

    # === Tab 4: 涨停数据 ===
    with tab4:
        st.markdown("### 🔥 市场连板高度监控")
        limit_df = fetch_limitup_data()
        
        if limit_df.empty:
            st.warning("⚠️ 交易所数据连接中...")
        else:
            limit_df.columns = limit_df.columns.astype(str)
            if '换手率' in limit_df.columns:
                if pd.api.types.is_numeric_dtype(limit_df['换手率']):
                    limit_df['换手率'] = limit_df['换手率'].apply(lambda x: f"{x*100:.1f}%")

            st.dataframe(
                limit_df,
                width='stretch', # 修复 Warning
                height=600,
                hide_index=True,
                column_config={
                    "股票名称": st.column_config.TextColumn("名称", width="small"),
                    "当前价": st.column_config.NumberColumn("现价", format="¥%.2f"),
                    "流通市值(亿)": st.column_config.NumberColumn("流通市值", format="%.1f亿"),
                    "连续涨停天数": st.column_config.NumberColumn("连板数", format="%d板"),
                    "买盘封单比": st.column_config.ProgressColumn("封单强度", format="%.2f", min_value=0, max_value=5),
                    "涨停原因": st.column_config.TextColumn("炒作概念", width="medium"),
                }
            )

    # === Tab 5: 形态选股 (Baostock 驱动) ===
    with tab5:
        st.markdown("### 📈 相似K线形态扫描")
        
        # 1. 数据库管理区域 (折叠)
        with st.expander("💾 本地数据库管理 (每日收盘后点此更新)", expanded=False):
            last_date = db_manager.get_last_update_time()
            st.write(f"当前数据库最新日期: **{last_date}**")
            
            if st.button("🚀 启动数据更新 (Baostock源)"):
                st.info("正在连接 Baostock 服务...")
                lg = bs.login()
                if lg.error_code != '0':
                    st.error(f"Baostock 登录失败: {lg.error_msg}")
                else:
                    try:
                        st.info("正在获取全市场股票列表...")
                        # 核心修复：自动回溯查找最近的交易日（解决周末0数据问题）
                        data_list = []
                        found_date = ""
                        # 尝试回溯最近 5 天
                        for delta in range(5):
                            check_date = (datetime.datetime.now() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
                            rs = bs.query_all_stock(day=check_date)
                            temp_list = []
                            while (rs.error_code == '0') & rs.next():
                                temp_list.append(rs.get_row_data())
                            
                            if len(temp_list) > 100: # 找到有效数据
                                data_list = temp_list
                                found_date = check_date
                                break
                        
                        if not data_list:
                             st.error("无法获取股票列表（最近5天均无数据），请检查 Baostock 服务。")
                        else:
                            st.success(f"成功获取股票列表 (交易日: {found_date})")
                            targets = []
                            for row in data_list:
                                code, name = row[0], row[2] # sh.600000, 浦发银行
                                # 过滤非A股和退市股
                                if (code.startswith("sh.6") or code.startswith("sz.0") or code.startswith("sz.3")) and "退" not in name:
                                    targets.append(code)

                            bs.logout() # 列表获取完毕先登出

                            st.write(f"准备更新 **{len(targets)}** 只股票数据...")
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            end_dt = datetime.datetime.now().strftime("%Y%m%d")
                            start_dt = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
                            
                            total = len(targets)
                            completed = 0
                            conn_write = sqlite3.connect(DB_NAME)
                            
                            # 启动多线程
                            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                                futures = {executor.submit(download_worker, sym, start_dt, end_dt): sym for sym in targets}
                                for future in concurrent.futures.as_completed(futures):
                                    df_res = future.result()
                                    if df_res is not None:
                                        df_res.to_sql(TABLE_NAME, conn_write, if_exists='append', index=False)
                                    completed += 1
                                    if completed % 20 == 0:
                                        progress_bar.progress(completed / total)
                                        status_text.text(f"已处理: {completed}/{total}")
                            
                            conn_write.close()
                            progress_bar.progress(1.0)
                            st.success(f"更新完成！成功入库股票数量：{completed}")

                    except Exception as e:
                        bs.logout()
                        st.error(f"发生错误: {e}")

        st.divider()

        # 2. 选股界面
        col_t1, col_t2 = st.columns([1, 2])
        
        with col_t1:
            st.subheader("1. 定义模板")
            tab_mode1, tab_mode2 = st.tabs(["股票模板", "手动输入"])
            
            with tab_mode1:
                input_code = st.text_input("股票代码 (如 sh.600519)", "sh.600624")
                date_range = st.date_input("时间段", value=(datetime.date(2020,1,6), datetime.date(2020,1,20)))
                
                if st.button("加载模板"):
                    s_d = date_range[0].strftime("%Y%m%d")
                    e_d = date_range[1].strftime("%Y%m%d")
                    # 使用 Baostock 获取单只股票作为模板
                    df_tpl = download_worker(input_code, s_d, e_d)
                    if df_tpl is not None and not df_tpl.empty:
                        st.session_state['target_pattern'] = df_tpl['close'].values
                        st.session_state['tpl_name'] = f"{input_code} ({s_d}-{e_d})"
                        st.success("模板加载成功")
                    else:
                        st.error("无法获取模板数据，请检查代码格式(需带sh./sz.)")

            with tab_mode2:
                manual = st.text_area("输入价格 (逗号分隔)", "10,11,12,11.5,13,14,13.5")
                if st.button("使用手动数据"):
                    try:
                        arr = [float(x) for x in manual.split(",")]
                        st.session_state['target_pattern'] = np.array(arr)
                        st.session_state['tpl_name'] = "手动序列"
                        st.success("手动序列已加载")
                    except:
                        st.error("格式错误")

        with col_t2:
            st.subheader("2. 模板预览")
            if 'target_pattern' in st.session_state:
                pat = st.session_state['target_pattern']
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=matcher.normalize(pat), mode='lines+markers', name='归一化形态'))
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), title=st.session_state.get('tpl_name'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👈 请先在左侧定义模板")

        # 3. 扫描执行
        st.subheader("3. 全市场扫描")
        threshold = st.slider("相似度阈值", 60, 95, 85)
        
        if st.button("开始匹配 (Search)", type="primary"):
            if 'target_pattern' not in st.session_state:
                st.warning("请先加载模板")
            else:
                target_seq = st.session_state['target_pattern']
                lookback = len(target_seq)
                all_symbols = db_manager.get_all_symbols()
                
                if not all_symbols:
                    st.error("数据库为空，请先在上方【数据库管理】中更新数据")
                else:
                    results = []
                    prog = st.progress(0)
                    for i, sym in enumerate(all_symbols):
                        df_p = db_manager.get_stock_data(sym, lookback)
                        if len(df_p) == lookback:
                            seq = df_p['close'].values
                            score = matcher.calculate_similarity(target_seq, seq)
                            if score >= threshold:
                                results.append({
                                    "代码": sym, "相似度": round(score, 2), 
                                    "现价": seq[-1], "data": seq
                                })
                        if i % 100 == 0: prog.progress((i+1)/len(all_symbols))
                    prog.progress(1.0)
                    
                    if results:
                        results.sort(key=lambda x: x['相似度'], reverse=True)
                        st.success(f"发现 {len(results)} 只相似股票")
                        
                        # 展示 Top 3 图表
                        cols = st.columns(min(3, len(results)))
                        norm_tgt = matcher.normalize(target_seq)
                        for i, col in enumerate(cols):
                            res = results[i]
                            with col:
                                st.caption(f"{res['代码']} (相似度:{res['相似度']}%)")
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(y=norm_tgt, name='模板', line=dict(dash='dash', color='gray')))
                                fig.add_trace(go.Scatter(y=matcher.normalize(res['data']), name='匹配', line=dict(color='red')))
                                fig.update_layout(showlegend=False, height=200, margin=dict(l=0,r=0,t=0,b=0))
                                st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(
                            pd.DataFrame(results).drop(columns=['data']), 
                            width='stretch' # 修复 Warning
                        )
                    else:
                        st.warning("未找到匹配股票")
    
    # === Tab 6: 善庄狙击 (Smart Money Sniper V5) ===
    with tab6:
        st.markdown("### 🎯 善庄狙击 V5 - 智能追踪高胜率游资")
        st.caption("基于龙虎榜数据，实时监控高胜率游资潜伏标的，多因子评分系统自动生成操作建议")
        
        # 检查数据库连接
        col_info1, col_info2 = st.columns([2, 1])
        with col_info1:
            st.info("📊 数据源：PostgreSQL龙虎榜数据库 (lhb_detail表) + 实时行情接口")
        with col_info2:
            if st.button("🔄 刷新狙击池", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        try:
            # 初始化狙击引擎
            with st.spinner("🔍 正在扫描善庄潜伏底仓..."):
                sniper = SmartMoneySniperV5()
                
                # 生成信号
                df_signals = sniper.generate_signals()
            
            if df_signals.empty:
                st.warning("⚠️ 当前无符合条件的标的。可能原因：")
                st.markdown("""
                - 数据库未配置或表结构不匹配
                - 最近5天无高胜率游资潜伏
                - 竞价金额过小（< 300万）
                - 评分未达到阈值（< 40分）
                """)
                
                # 显示连接状态
                with st.expander("🔧 数据库诊断信息"):
                    st.code(f"""
数据库主机: {sniper.db_host}
数据库名称: {sniper.db_name}
数据库用户: {sniper.db_user}
连接状态: {'✅ 已连接' if sniper.engine else '❌ 未连接'}
游资缓存: {len(sniper.broker_cache)} 个活跃席位
                    """)
            else:
                # 显示统计信息
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("🎯 狙击标的", f"{len(df_signals)} 只")
                with col_stat2:
                    top_score = df_signals['分数'].max() if not df_signals.empty else 0
                    st.metric("⭐ 最高评分", f"{top_score} 分")
                with col_stat3:
                    top_win_rate = df_signals['庄家最高胜率'].max() if not df_signals.empty else 0
                    st.metric("🏆 最高胜率", f"{top_win_rate}%")
                with col_stat4:
                    active_brokers = len(sniper.broker_cache)
                    st.metric("👥 活跃游资", f"{active_brokers} 席")
                
                st.divider()
                
                # 高亮显示首推标的
                if len(df_signals) > 0:
                    st.markdown("#### 💎 首推标的（按胜率+评分排序）")
                    top_pick = df_signals.iloc[0]
                    
                    col_top1, col_top2 = st.columns([1, 2])
                    with col_top1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 20px; border-radius: 15px; color: white;">
                            <h2 style="margin:0; color:white;">{top_pick['名称']}</h2>
                            <p style="font-size: 1.5rem; margin: 5px 0; color:white;">{top_pick['代码']}</p>
                            <p style="font-size: 1.2rem; margin: 5px 0; color:white;">
                                涨幅: <strong>{top_pick['涨幅%']}%</strong>
                            </p>
                            <p style="font-size: 0.9rem; opacity: 0.9; color:white;">
                                {top_pick['行业']} | {top_pick['流通市值(亿)']}亿
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_top2:
                        st.markdown(f"""
                        **🎯 综合评分:** {top_pick['分数']} 分
                        **🏆 庄家胜率:** {top_pick['庄家最高胜率']}%
                        **💰 竞价金额:** {top_pick['竞价额']} 万
                        
                        **潜伏庄家:**
                        {top_pick['潜伏庄家']}
                        
                        **📍 买入计划:**
                        {top_pick['买入计划']}
                        
                        **🚪 T+1卖出:**
                        {top_pick['T+1卖出']}
                        """)
                
                st.divider()
                
                # 完整列表
                st.markdown("#### 📋 完整狙击池")
                
                # 格式化显示
                display_df = df_signals[['代码', '名称', '潜伏庄家', '庄家最高胜率', '分数',
                                         '流通市值(亿)', '行业', '涨幅%', '竞价额', '买入计划', 'T+1卖出']]
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500,
                    column_config={
                        "代码": st.column_config.TextColumn("代码", width="small"),
                        "名称": st.column_config.TextColumn("名称", width="small"),
                        "潜伏庄家": st.column_config.TextColumn("潜伏庄家", width="medium"),
                        "庄家最高胜率": st.column_config.NumberColumn("胜率%", format="%d%%"),
                        "分数": st.column_config.ProgressColumn("评分", min_value=0, max_value=100, format="%d"),
                        "流通市值(亿)": st.column_config.NumberColumn("市值", format="%.1f亿"),
                        "行业": st.column_config.TextColumn("行业", width="small"),
                        "涨幅%": st.column_config.NumberColumn("涨幅", format="%.2f%%"),
                        "竞价额": st.column_config.NumberColumn("竞价额", format="%d万"),
                        "买入计划": st.column_config.TextColumn("买入计划", width="large"),
                        "T+1卖出": st.column_config.TextColumn("T+1卖出", width="medium"),
                    }
                )
                
                # 评分说明
                with st.expander("📖 评分规则说明"):
                    st.markdown("""
                    **多因子评分体系 (满分100+):**
                    
                    1. **竞价金额因子** (最高20分)
                       - 竞价额 > 3000万: +20分
                       - 竞价额 > 1000万: +10分
                    
                    2. **竞价换手率** (最高20分)
                       - 换手率 > 0.8%: +20分
                    
                    3. **量比因子** (最高15分)
                       - 量比 > 5: +15分
                       - 量比 > 2: +10分
                    
                    4. **涨幅因子** (最高25分)
                       - 1% ≤ 涨幅 ≤ 5%: +25分 (黄金区间)
                       - 5% < 涨幅 < 8% 且竞价额>2000万: +15分
                       - 涨幅 < -2%: -20分 (风险)
                    
                    5. **庄家胜率因子** (最高20分)
                       - 胜率 ≥ 80%: +20分
                       - 胜率 ≥ 60%: +10分
                    
                    6. **流通市值因子** (5分)
                       - 市值 < 30亿: +5分 (灵活性溢价)
                    
                    **操作建议:**
                    - **80分以上**: 激进介入，低吸为主
                    - **60-80分**: 稳健参与，等回调
                    - **40-60分**: 观察为主，破位止损
                    """)
                
                # 游资排行榜
                with st.expander("🏆 活跃游资胜率排行 (近60天)"):
                    broker_df = sniper.get_broker_ranking_data()
                    if not broker_df.empty:
                        st.dataframe(
                            broker_df.head(20),
                            use_container_width=True,
                            column_config={
                                "dept_name": "席位名称",
                                "操作次数": st.column_config.NumberColumn("操作次数", format="%d次"),
                                "胜率": st.column_config.ProgressColumn("T+2胜率", min_value=0, max_value=100, format="%d%%"),
                            }
                        )
                    else:
                        st.warning("无法加载游资数据")
        
        except Exception as e:
            st.error(f"❌ 善庄狙击引擎启动失败: {str(e)}")
            st.markdown("""
            **可能的解决方案:**
            1. 检查 `.env` 文件中的数据库配置
            2. 确保 PostgreSQL 数据库可访问
            3. 验证 `lhb_detail` 表是否存在且有数据
            4. 检查网络连接和防火墙设置
            """)
            
            with st.expander("🔍 详细错误信息"):
                import traceback
                st.code(traceback.format_exc())
    
    render_simulation_tab(tab7)
def render_simulation_tab(tab6):
    with tab6:
        st.markdown("### 🤖 股票/期货 日内网格交易仿真 (Pro版)")
        st.caption("集成新浪高速行情接口，支持期货主力连续指标映射。")
        
        col_conf, col_run = st.columns([1, 2])

        with col_conf:
            with st.container(border=True):
                st.subheader("🛠️ 策略配置")
                
                sim_type = st.radio("品种类型", ["股票", "期货"], horizontal=True)
                
                if sim_type == "股票":
                    sim_code = st.text_input("代码", "000063", help="无需后缀")
                    sim_multiplier = 1
                    sim_margin = 1.0
                    sim_amt = st.number_input("单笔金额", 50000)
                else:
                    sim_code = st.text_input("期货代码", "RB2505", help="例如: RB2505, IM2506")
                    c1, c2 = st.columns(2)
                    sim_multiplier = c1.number_input("合约乘数", 10, help="RB=10, IF=300")
                    sim_margin = c2.number_input("保证金率", 0.1, step=0.01)
                    sim_amt = st.number_input("单笔保证金", 5000)
                
                sim_cash = st.number_input("初始资金", 500000)
                sim_duration = st.slider("运行时长(分)", 10, 360, 60)
                
                start_btn = st.button("🚀 开始仿真", type="primary", use_container_width=True)

        with col_run:
            if start_btn:
                st.success(f"正在连接行情源... 目标: {sim_code}")
                
                # UI 占位符
                status_box = st.empty()
                log_box = st.empty()
                
                # 配置
                cfg = {
                    'symbol': sim_code,
                    'asset_type': sim_type,
                    'initial_cash': sim_cash,
                    'grid_amt': sim_amt,
                    'multiplier': sim_multiplier,
                    'margin_rate': sim_margin,
                    'adx_threshold': 30,
                    'default_price': 3000 # 兜底用
                }
                
                # 实例化引擎
                engine = GridTraderEngine(cfg, log_box, status_box)
                
                # 运行循环
                end_time = time.time() + sim_duration * 60
                prog_bar = st.progress(0)
                
                try:
                    while time.time() < end_time:
                        # 1. 获取价格
                        if sim_type == '股票':
                            p = get_stock_price_realtime(sim_code)
                        else:
                            p = get_futures_price_robust(sim_code) # 使用修复后的新浪接口
                        
                        # 2. 驱动引擎
                        if p:
                            engine.run_step(p)
                        else:
                            # 仅在日志中显示连接状态，不阻塞
                            # engine.add_log("行情连接中...") 
                            pass
                        
                        # 3. 进度条
                        remain = end_time - time.time()
                        total = sim_duration * 60
                        prog_bar.progress(1.0 - max(0, remain / total))
                        
                        time.sleep(2) # 2秒刷新一次
                        
                    st.success("仿真结束")
                    
                except Exception as e:
                    st.error(f"运行中断: {e}")
            else:
                st.info("👈 请在左侧输入参数，点击【开始仿真】")
                st.markdown("""
                **功能说明:**
                1. **高速接口**: 使用 HTTP 直连新浪期货接口，解决 AkShare 盘中卡顿问题。
                2. **智能映射**: 输入 `RB2505`，系统会自动获取 `RB0` (主力连续) 的历史 K 线来计算 ATR 和 ADX 指标。
                3. **实时风控**: 
                   - **多头趋势 (ADX>30 & +DI>-DI)**: 暂停卖出，防止卖飞。
                   - **空头趋势 (ADX>30 & -DI>+DI)**: 暂停买入，防止接飞刀。
                """)

if __name__ == "__main__":
    main()