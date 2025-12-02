"""
测试 python-okx SDK 的正确导入方式
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("测试 python-okx SDK 导入")
logger.info("=" * 60)

# 测试方式 1: 直接导入模块
try:
    logger.info("\n方式 1: 直接导入模块")
    import okx
    logger.info(f"✅ okx 模块导入成功")
    logger.info(f"   okx 模块内容: {dir(okx)}")
except Exception as e:
    logger.error(f"❌ 导入失败: {e}")

# 测试方式 2: 导入子模块
try:
    logger.info("\n方式 2: 导入子模块")
    import okx.Account as Account
    import okx.MarketData as MarketData
    import okx.Trade as Trade
    import okx.PublicData as PublicData
    logger.info(f"✅ 子模块导入成功")
    logger.info(f"   Account 类型: {type(Account)}")
    logger.info(f"   Account 内容: {dir(Account)}")
except Exception as e:
    logger.error(f"❌ 导入失败: {e}")

# 测试方式 3: 查找正确的 API 类
try:
    logger.info("\n方式 3: 查找 API 类")
    import okx.Account as Account
    
    # 检查是否有 AccountAPI 类
    if hasattr(Account, 'AccountAPI'):
        logger.info(f"✅ 找到 AccountAPI 类")
        logger.info(f"   AccountAPI 类型: {type(Account.AccountAPI)}")
    else:
        logger.info(f"⚠️ 未找到 AccountAPI 类")
        logger.info(f"   Account 模块内容: {[x for x in dir(Account) if not x.startswith('_')]}")
        
except Exception as e:
    logger.error(f"❌ 测试失败: {e}")

# 测试方式 4: 尝试实例化
try:
    logger.info("\n方式 4: 尝试实例化 API 类")
    import okx.Account as Account
    
    if hasattr(Account, 'AccountAPI'):
        # 尝试创建实例（使用测试参数）
        api = Account.AccountAPI(
            api_key='test',
            api_secret_key='test',
            passphrase='test',
            flag='0'
        )
        logger.info(f"✅ AccountAPI 实例化成功: {type(api)}")
    else:
        logger.warning("⚠️ AccountAPI 类不存在")
        
except Exception as e:
    logger.error(f"❌ 实例化失败: {e}")
    import traceback
    logger.error(traceback.format_exc())

logger.info("\n" + "=" * 60)
logger.info("测试完成")
logger.info("=" * 60)
