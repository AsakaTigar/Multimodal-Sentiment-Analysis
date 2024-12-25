import os  # 导入操作系统相关库，用于环境变量设置等操作
import torch  # 导入 PyTorch 库，用于深度学习模型的构建与训练
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条


from opts import *  # 从 opts 模块导入配置选项
from dataset import MMDataLoader  # 导入 MMDataLoader 用于加载数据
from utils import AverageMeter  # 导入 AverageMeter 类用于计算和记录损失
from almt import build_model  # 导入构建模型的函数
from metric import MetricsTop  # 导入用于计算评估指标的函数


# 解析命令行选项并赋值给 opt
opt = parse_opts()

# 设置 CUDA 可见设备，即指定使用哪些 GPU 进行计算
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES

# 判断是否可以使用 GPU
USE_CUDA = torch.cuda.is_available()

# 如果 GPU 可用，使用 GPU，否则使用 CPU
device = torch.device("cuda" if USE_CUDA else "cpu")

# 打印设备信息，显示当前使用的设备和 CUDA 设备编号
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))

# 初始化训练和验证的 MAE（Mean Absolute Error）列表
train_mae, val_mae = [], []

# 主函数
def main():
    opt = parse_opts()  # 再次解析命令行参数

    model = build_model(opt).to(device)  # 构建模型并将其移动到指定的设备（GPU/CPU）
    model.load_state_dict(torch.load(opt.test_checkpoint)['state_dict'])  # 加载模型的预训练权重

    dataLoader = MMDataLoader(opt)  # 加载数据集

    loss_fn = torch.nn.MSELoss()  # 定义损失函数为均方误差（MSE）
    metrics = MetricsTop().getMetics(opt.datasetName)  # 获取评估指标，基于数据集的不同选择不同的指标

    # 开始测试模型
    test(model, dataLoader['test'], loss_fn, metrics)

# 测试函数
def test(model, test_loader, loss_fn, metrics):
    test_pbar = tqdm(enumerate(test_loader))  # 使用 tqdm 显示测试进度条

    losses = AverageMeter()  # 创建 AverageMeter 实例，用于计算和记录损失
    y_pred, y_true = [], []  # 用于保存预测值和真实值的列表

    model.eval()  # 将模型设置为评估模式，关闭 dropout 和 batch normalization
    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        for cur_iter, data in test_pbar:  # 遍历测试集数据
            # 获取当前批次的数据，包括视觉、音频、文本数据
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)  # 获取标签
            label = label.view(-1, 1)  # 将标签维度调整为（batch_size, 1）
            batchsize = img.shape[0]  # 获取当前批次的大小

            output = model(img, audio, text)  # 通过模型进行预测，获取输出结果

            loss = loss_fn(output, label)  # 计算当前批次的损失

            # 将当前批次的预测值和真实值分别添加到列表中
            y_pred.append(output.cpu())  # 将预测结果移到 CPU
            y_true.append(label.cpu())   # 将真实标签移到 CPU

            # 更新损失记录
            losses.update(loss.item(), batchsize)

            # 更新进度条的描述和后缀信息，显示当前损失值
            test_pbar.set_description('test')
            test_pbar.set_postfix({'loss': '{:.5f}'.format(losses.value_avg)})

        # 将所有批次的预测结果和真实标签拼接起来
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # 计算评估指标
        test_results = metrics(pred, true)
        # 打印测试结果
        print(test_results)

# 主程序入口
if __name__ == '__main__':
    main()  # 调用主函数
