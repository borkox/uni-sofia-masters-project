import os, sys, getopt, glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np


title = 'Резултати от обучение'
ylabel = "получена награда"
xlabel = "номер на епизод"
scores_file = "outputs/scores.txt"
output_image_filename = None

opts, args = getopt.getopt(sys.argv[1:],"hx:y:i:o:")
print (f"Script {sys.argv[0]} started with options: ", opts)
for opt, arg in opts:
    if opt in ("-h"):
        print ('USAGE:> plot_scores.py OPTIONS')
        print ('OPTIONS: ')
        print ('   -x <xlabel>')
        print ('')
        print ('   -y <ylabel>')
        print ('')
        print ('   -i <input_scores_file>')
        print ('')
        print ('   -o <output_image_png_file> , if set will save image, otherwise shows the image.')
        sys.exit()
    elif opt in ("-x"):
        xlabel = arg
    elif opt in ("-y"):
        ylabel = arg
    elif opt in ("-i"):
        scores_file = arg
    elif opt in ("-o"):
        output_image_filename = arg
    else:
        print("Unknown option:", opt)
        sys.exit()


sns.set()

scores = np.loadtxt(scores_file)

plt.plot(scores)
plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.title(title)

reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))
plt.plot(y_pred)
if output_image_filename is not None:
    plt.savefig(f'{output_image_filename}', format='png')
else:
    plt.show()