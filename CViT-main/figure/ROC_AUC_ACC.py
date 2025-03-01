import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# 加载标签文件和预测文件
metadata_path = r"E:\Daima\dataset\DFDC\video\test\metadata.json"
predictions_path = r'E:\Daima\face_fake\CViT-main\wprediction\4090RepBn8_dfdc_ff_total_test_dfdc.csv'

# 加载JSON标签文件
metadata = pd.read_json(metadata_path)

# 加载CSV预测文件
predictions = pd.read_csv(predictions_path)

# 过滤掉标签等于5的预测数据
predictions = predictions[predictions['label'] != 0.5]

# 将JSON标签转换为字典形式
true_labels_dict = metadata.T[['label']].to_dict()['label']

# 转换标签为二进制形式：REAL为0，FAKE为1
binary_true_labels = predictions['filename'].apply(lambda x: 1 if true_labels_dict.get(x, "REAL") == "FAKE" else 0)
binary_predictions = predictions['label'].apply(lambda x: 1 if x > 0.5 else 0)

# 计算准确率
accuracy = accuracy_score(binary_true_labels, binary_predictions)

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(binary_true_labels, predictions['label'])
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print(f"Accuracy: {accuracy:.5f}")
print(f"AUC: {roc_auc:.5f}")
