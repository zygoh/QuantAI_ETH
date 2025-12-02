"""
测试 OKX SDK 集成

验证 python-okx SDK 是否正确安装和配置
"""
import asyncio
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_okx_sdk():
    """测试 OKX SDK 集成"""
    try:
        # 1. 测试 SDK 导入
        logger.info("=" * 60)
        logger.info("测试 1: 验证 python-okx SDK 导入")
        logger.info("=" * 60)
        
        try:
            from okx import Account, MarketData, Trade, PublicData
            from okx.exceptions import OkxAPIException, OkxRequestException, OkxParamsException
            logger.info("✅ python-okx SDK 导入成功")
        except ImportError as e:
            logger.error(f"❌ python-okx SDK 导入失败: {e}")
            logger.error("   请运行: pip install python-okx==0.4.0")
            return False
        
        # 2. 测试 OKXClient 导入
        logger.info("\n" + "=" * 60)
        logger.info("测试 2: 验证 OKXClient 导入")
        logger.info("=" * 60)
        
        try:
            from app.exchange.okx_client import OKXClient
            logger.info("✅ OKXClient 导入成功")
        except ImportError as e:
            logger.error(f"❌ OKXClient 导入失败: {e}")
            return False
        
        # 3. 测试 OKXClient 初始化
        logger.info("\n" + "=" * 60)
        logger.info("测试 3: 验证 OKXClient 初始化")
        logger.info("=" * 60)
        
        try:
            # 使用测试配置（不需要真实的 API 密钥）
            test_config = {
                'api_key': 'test_key',
                'secret_key': 'test_secret',
                'passphrase': 'test_passphrase'
            }
            
            client = OKXClient(test_config)
            logger.info("✅ OKXClient 初始化成功")
            logger.info(f"   - SDK Account API: {client.account_api is not None}")
            logger.info(f"   - SDK Market API: {client.market_api is not None}")
            logger.info(f"   - SDK Trade API: {client.trade_api is not None}")
            logger.info(f"   - SDK Public API: {client.public_api is not None}")
            
        except Exception as e:
            logger.error(f"❌ OKXClient 初始化失败: {e}")
            return False
        
        # 4. 测试方法存在性
        logger.info("\n" + "=" * 60)
        logger.info("测试 4: 验证 OKXClient 方法")
        logger.info("=" * 60)
        
        required_methods = [
            'test_connection',
            'get_server_time',
            'get_klines',
            'get_ticker_price',
            'get_account_info',
            'get_position_info',
            'place_order',
            'cancel_order',
            'get_open_orders',
            'change_leverage',
            '_handle_sdk_exception'
        ]
        
        for method_name in required_methods:
            if hasattr(client, method_name):
                logger.info(f"✅ 方法存在: {method_name}")
            else:
                logger.error(f"❌ 方法缺失: {method_name}")
                return False
        
        # 5. 测试异常处理
        logger.info("\n" + "=" * 60)
        logger.info("测试 5: 验证 SDK 异常处理")
        logger.info("=" * 60)
        
        try:
            # 测试 API 异常转换
            test_exception = OkxAPIException(code='50011', message='Rate limit exceeded')
            try:
                client._handle_sdk_exception(test_exception)
            except Exception as e:
                logger.info(f"✅ SDK 异常转换正常: {type(e).__name__}")
            
            # 测试请求异常转换
            test_exception = OkxRequestException('Network error')
            try:
                client._handle_sdk_exception(test_exception)
            except Exception as e:
                logger.info(f"✅ SDK 异常转换正常: {type(e).__name__}")
            
        except Exception as e:
            logger.error(f"❌ 异常处理测试失败: {e}")
            return False
        
        # 测试完成
        logger.info("\n" + "=" * 60)
        logger.info("✅ 所有测试通过！")
        logger.info("=" * 60)
        logger.info("\n📝 总结:")
        logger.info("  1. python-okx SDK 已正确安装")
        logger.info("  2. OKXClient 使用 SDK 进行初始化")
        logger.info("  3. 所有必需方法已实现")
        logger.info("  4. SDK 异常处理正常工作")
        logger.info("\n🎉 OKX SDK 集成成功！")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    result = asyncio.run(test_okx_sdk())
    exit(0 if result else 1)
