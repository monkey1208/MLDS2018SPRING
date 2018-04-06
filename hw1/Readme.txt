python 1-1_fit_function.py
#可-h看可調整參數內容
#主要包含 	-train 可以default的設定訓練模型
#		-test  需要 shallow, deep, superdeep三個模型(這個功能主要是作圖用)
#		-deep  可以用deep的模型訓練
#會訓練出對 f(x) = sgn(sin(x))作fit的模型 需要以 1-1_GenerateData.py [data_path] [sample數]先做出訓練data
#並存取loss的紀錄進入一個csv檔

python 1-1_MNIST.py [#epoch] [#model數] [0 or 1]
#會自行下載 MNIST dataset argv[3] 定義使用哪個model (0 for shallow; 1 for deep)
#epoch定義要train幾個epoch
#model定義要train幾個model (1-2作圖用)
#會生出一個csv檔 內容包含model參數降維後的結果及loss和accuracy的紀錄

#python 1-2_plot_optimization.py [#csv] [#model數]
#csv 上面那個code生出的csv檔
#上面所使用的model數

python 1-2_cal_err_surface.py [#epoch]
#epoch定義再換成second order optimizer之前要先以gradient descending訓練多少epoch
#會生出一個csv檔包含loss值及TSNE將為過後的weight sample共5010筆(5000筆sample及10筆training 過程的紀錄)

python 1-2_plot_err_surface.py [#data_path]
#data_path 為上面生出檔案的路徑

python3 hw1-3-1.py
python3 plot.py 1_3_1

python3 hw1-3-2.py
python3 plot.py 1_3_2

python3 hw1-3-3-1.py
python3 plot.py 1_3_3_1

python3 hw1-3-3-2.py
python3 plot.py 1_3_3_2