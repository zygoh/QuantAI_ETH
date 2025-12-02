"""
OKX API认证诊断脚本

用于测试OKX API密钥配置是否正确
"""
import hmac
import hashlib
import base64
import requests
from datetime import datetime
from app.core.config import settings

def test_okx_authentication():
    """测试OKX API认证"""
    
    print("=" * 60)
    print("OKX API认证诊断")
    print("=" * 60)
    
    # 1. 检查配置
    print("\n1. 检查配置信息:")
    print(f"   API Key: {settings.OKX_API_KEY[:8]}...{settings.OKX_API_KEY[-4:]}")
    print(f"   Secret Key: {settings.OKX_SECRET_KEY[:8]}...{settings.OKX_SECRET_KEY[-4:]}")
    print(f"   Passphrase: {'*' * len(settings.OKX_PASSPHRASE)}")
    print(f"   使用代理: {settings.USE_PROXY}")
    if settings.USE_PROXY:
        print(f"   代理地址: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
        print(f"   代理类型: {settings.PROXY_TYPE}")
    
    # 2. 测试公共接口（无需认证）
    print("\n2. 测试公共接口（无需认证）:")
    try:
        proxies = None
        if settings.USE_PROXY:
            proxy_url = f"{settings.PROXY_TYPE}://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            proxies = {"http": proxy_url, "https": proxy_url}
        
        url = "https://www.okx.com/api/v5/public/time"
        response = requests.get(url, proxies=proxies, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') == '0':
            server_time = data['data'][0]['ts']
            print(f"   ✅ 公共接口访问成功")
            print(f"   服务器时间: {server_time}")
        else:
            print(f"   ❌ 公共接口返回错误: {data}")
    except Exception as e:
        print(f"   ❌ 公共接口访问失败: {e}")
        return
    
    # 3. 测试签名生成
    print("\n3. 测试签名生成:")
    try:
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
        method = "GET"
        request_path = "/api/v5/account/balance"
        body = ""
        
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(settings.OKX_SECRET_KEY, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        signature = base64.b64encode(mac.digest()).decode()
        
        print(f"   时间戳: {timestamp}")
        print(f"   签名消息: {message}")
        print(f"   签名结果: {signature[:20]}...")
        print(f"   ✅ 签名生成成功")
    except Exception as e:
        print(f"   ❌ 签名生成失败: {e}")
        return
    
    # 4. 测试私有接口（需要认证）
    print("\n4. 测试私有接口（需要认证）:")
    try:
        headers = {
            'OK-ACCESS-KEY': settings.OKX_API_KEY,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': settings.OKX_PASSPHRASE,
            'Content-Type': 'application/json'
        }
        
        url = f"https://www.okx.com{request_path}"
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        
        print(f"   HTTP状态码: {response.status_code}")
        print(f"   响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   响应数据: {data}")
            
            if data.get('code') == '0':
                print(f"   ✅ 私有接口访问成功")
                print(f"   账户余额: {data.get('data', [])}")
            else:
                print(f"   ❌ API返回错误:")
                print(f"      错误码: {data.get('code')}")
                print(f"      错误信息: {data.get('msg')}")
                print_error_solutions(data.get('code'))
        else:
            print(f"   ❌ HTTP请求失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            print_http_error_solutions(response.status_code)
            
    except Exception as e:
        print(f"   ❌ 私有接口访问失败: {e}")
        import traceback
        print(f"   详细错误: {traceback.format_exc()}")

def print_error_solutions(error_code):
    """打印错误解决方案"""
    solutions = {
        '50100': [
            "API Key不正确",
            "解决方案：检查.env文件中的OKX_API_KEY是否正确"
        ],
        '50101': [
            "API Key已过期或被删除",
            "解决方案：在OKX官网重新创建API Key"
        ],
        '50102': [
            "时间戳错误",
            "解决方案：同步系统时间（Windows: w32tm /resync）"
        ],
        '50103': [
            "请求头不正确",
            "解决方案：检查请求头格式是否符合OKX要求"
        ],
        '50104': [
            "Passphrase不正确",
            "解决方案：检查.env文件中的OKX_PASSPHRASE是否正确"
        ],
        '50105': [
            "签名不正确",
            "解决方案：检查.env文件中的OKX_SECRET_KEY是否正确"
        ],
        '50111': [
            "IP不在白名单中",
            "解决方案：在OKX官网API管理中添加当前IP到白名单，或移除IP限制"
        ],
        '50113': [
            "API Key权限不足",
            "解决方案：在OKX官网API管理中启用'读取'和'合约交易'权限"
        ]
    }
    
    if error_code in solutions:
        print(f"\n   💡 可能的原因和解决方案:")
        for line in solutions[error_code]:
            print(f"      {line}")

def print_http_error_solutions(status_code):
    """打印HTTP错误解决方案"""
    if status_code == 401:
        print(f"\n   💡 401 Unauthorized 常见原因:")
        print(f"      1. API Key、Secret Key 或 Passphrase 不正确")
        print(f"      2. API Key权限不足（需要启用'读取'和'合约交易'权限）")
        print(f"      3. IP不在白名单中")
        print(f"      4. 签名算法错误")
        print(f"      5. 时间戳不同步")
        print(f"\n   🔧 解决步骤:")
        print(f"      1. 登录OKX官网 → 个人中心 → API")
        print(f"      2. 检查API Key是否存在且未过期")
        print(f"      3. 确认已启用'读取'和'合约交易'权限")
        print(f"      4. 检查IP白名单设置（建议暂时移除IP限制测试）")
        print(f"      5. 同步系统时间: w32tm /resync")
    elif status_code == 403:
        print(f"\n   💡 403 Forbidden 常见原因:")
        print(f"      1. API Key权限不足")
        print(f"      2. IP被封禁")
        print(f"      3. 账户被限制")

if __name__ == "__main__":
    test_okx_authentication()
