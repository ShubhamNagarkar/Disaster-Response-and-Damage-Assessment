from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np

REL = os.getcwd()
VISUALIZATION = REL + '/visualization/resnet50/'
import seaborn as sns


def draw_training_curves(train_losses, test_losses, curve_name, epoch):
    plt.clf()
    max_y = 2.0
    if curve_name == "Accuracy":
        max_y = 1.0
        plt.ylim([0, max_y])

    plt.xlim([0, epoch])
    plt.xticks(np.arange(0, epoch, 3))
    plt.plot(train_losses, label='Training {}'.format(curve_name))
    plt.plot(test_losses, label='Val {}'.format(curve_name))
    plt.ylabel(curve_name)
    plt.xlabel('Epoch')
    plt.legend(frameon=False)
    plt.savefig("Loss_curve_Late_Fusion.png".format(curve_name))
    plt.close()


def plot_cm(lab, pred):
    target_names = ['A', 'I', 'N', 'O', 'R']

    cm = confusion_matrix(lab, pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap="YlGnBu")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout(h_pad=5.0)
    plt.savefig("late-fusion-confusion_matrix.png")

    plt.close()


# x11 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# x12 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 3, 3, 3, 3, 3, 5, 2, 1, 1]
# x21 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2]
# x22 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 3, 2,2]
# x31 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# x32 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 2, 3]
# x41 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
# x42 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2]
# x51 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
# x52 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3]
# lab = x11 + x21 + x31 + x41 + x51
# pred = x12 + x22 + x32 + x42 + x52
# print(len(pred))
# print(len(lab))
#
# plot_cm(lab, pred)


train = [1.2, 1.25, 1.15, 1.10, 1.13, 1.0, .95, .9, .86, .81, .77, .74, .71, .68, .63, .57, .54, .51, .46, .42, .39,
         .39, .39, .38, .38]

val = [1.4, 1.35, 1.34, 1.30, 1.30, 1.24, 1.20, 1.22, 1.02, .97, .92, .87, .81, .78, .71, .66, .62, .57, .54, .52, .49,
       .49, .49, .48, .48]
#
print(len(train))
print(len(val))
# print(train)

#
# print(len(val))
#
draw_training_curves(train, val, curve_name="Loss", epoch=25)
