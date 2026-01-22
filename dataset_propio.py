import pandas as pd
import numpy as np

def main():
    x1_values, x2_values, x3_values, y_values = [], [], [], []
    for i in range(500):
        x1_values.append(np.random.random()*100)
        x2_values.append(np.random.random()*100)
        x3_values.append(np.random.random()*100)
        y_values.append(np.sin(x1_values[i])+np.sin(x2_values[i]))
    df = pd.DataFrame({'date': range(1,501), 'x1':x1_values, 'x2':x2_values, 'x3':x3_values, 'y':y_values})
    df.to_csv("input_csv_files/own_dataset.csv")
if __name__ == '__main__':
    main()