python 1-1_fit_function.py
#�i-h�ݥi�վ�ѼƤ��e
#�D�n�]�t 	-train �i�Hdefault���]�w�V�m�ҫ�
#		-test  �ݭn shallow, deep, superdeep�T�Ӽҫ�(�o�ӥ\��D�n�O�@�ϥ�)
#		-deep  �i�H��deep���ҫ��V�m
#�|�V�m�X�� f(x) = sgn(sin(x))�@fit���ҫ� �ݭn�H 1-1_GenerateData.py [data_path] [sample��]�����X�V�mdata
#�æs��loss�������i�J�@��csv��

python 1-1_MNIST.py [#epoch] [#model��] [0 or 1]
#�|�ۦ�U�� MNIST dataset argv[3] �w�q�ϥέ���model (0 for shallow; 1 for deep)
#epoch�w�q�ntrain�X��epoch
#model�w�q�ntrain�X��model (1-2�@�ϥ�)
#�|�ͥX�@��csv�� ���e�]�tmodel�Ѽƭ����᪺���G��loss�Maccuracy������

#python 1-2_plot_optimization.py [#csv] [#model��]
#csv �W������code�ͥX��csv��
#�W���ҨϥΪ�model��

python 1-2_cal_err_surface.py [#epoch]
#epoch�w�q�A����second order optimizer���e�n���Hgradient descending�V�m�h��epoch
#�|�ͥX�@��csv�ɥ]�tloss�Ȥ�TSNE�N���L�᪺weight sample�@5010��(5000��sample��10��training �L�{������)

python 1-2_plot_err_surface.py [#data_path]
#data_path ���W���ͥX�ɮת����|

python3 hw1-3-1.py
python3 plot.py 1_3_1

python3 hw1-3-2.py
python3 plot.py 1_3_2

python3 hw1-3-3-1.py
python3 plot.py 1_3_3_1

python3 hw1-3-3-2.py
python3 plot.py 1_3_3_2