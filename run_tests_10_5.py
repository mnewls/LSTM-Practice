import attempted_lstm_9_18 as pwr_ball_pred
import time

# 1/69 = 1.449275362318841

#res_1 = pwr_ball_pred.main(69, 69, 6, .0001, 1, 1)
#2.6345933562428407
this_list = []

#layer_one_dep, layer_two_dep, layer_three_dep, sample_len, learn_rate, epoch_num, batch_num

res_1 = pwr_ball_pred.main(207,207,20,.0000001,20,1024)
print(res_1)
this_list.append(res_1)
time.sleep((10*60))
#1.8626309662398137

res_2 = pwr_ball_pred.main(69,138,30,.000001,50,512)
print(res_2)
this_list.append(res_2)
time.sleep((10*60))
#1.6489988221436984

res_3 = pwr_ball_pred.main(69,207,24,.00000001,100,512)
print(res_3)
this_list.append(res_3)
time.sleep((10*60))
#1.5204678362573099

res_4 = pwr_ball_pred.main(69,276,60,.0000001,5,512)
print(res_4)
this_list.append(res_4)
time.sleep((10*60))
#1.8315018315018317

res_5 = pwr_ball_pred.main(276,276,18,.00005,2,1024)
print(res_5)
this_list.append(res_5)
time.sleep((10*60))
#2.5551684088269457

res_6 = pwr_ball_pred.main(138,207,18,.000005,50,512)
print(res_6)
this_list.append(res_6)
time.sleep((10*60))
#1.3937282229965158

res_7 = pwr_ball_pred.main(69,138,12,.000002,20,512)
print(res_7)
this_list.append(res_7)
time.sleep((10*60))

res_8 = pwr_ball_pred.main(138,207,18,.0005,2,2)
print(res_8)
this_list.append(res_8)
print(*this_list)