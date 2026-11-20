# particle-emitter

## 题面

爆炸特效一上，粒子不是不发射就是瞬间消失：发射按 `rate*dt` 取整，低速率永远不发粒子；不累积余数，60 fps 下速率误差 30%；积分用显式欧拉，重力大时粒子飞出；阻力按帧数衰减，帧率变化时手感不同；容量满时丢弃新粒子，爆炸效果缺一块；排序用对象地址，顺序不稳定；随机用 `random`，回放不可复现；overdraw 不限制，粒子密集时掉帧；生命周期曲线端点不为 0，粒子消失瞬间闪一下；批量共用粒子池，两套效果互相污染。particles/emitter.py 是核心，tests/test_emitter.py 15个红。要修的是：让 emit 按小数累积发射、用半隐式欧拉并按秒施加指数阻力、recycle 保留最新粒子、按相机距离稳定地从远到近排序、用显式 seed 的确定性哈希生成速度、按体积裁剪 overdraw、让生命周期曲线两端严格为 1 和 0，并让 simulate_batch 为每套配置使用独立状态且任一配置非法时整体回滚。

## 运行方式

```bash
cd D:\gsb-20261119\particle-emitter
python -m pytest tests -q
```

初始状态应恰好输出：

```text
15 failed
```

15 条测试全部因桩函数 `raise NotImplementedError` 而失败；收集、导入和语法检查必须通过，不能出现 `error`。实现后测试应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 测试约束

`tests/test_emitter.py` 是验收测试，不得修改、删除或新增；修复只能落在 `particles/emitter.py`。只使用 Python 标准库，离线运行，不使用全局随机状态、不 sleep。所有位置断言允许 `1e-6` 容差；生命周期和 `dt` 都以秒为单位；`capacity` 是保留粒子的硬上限；粒子的初始速度必须由显式传入的 `seed` 决定。
