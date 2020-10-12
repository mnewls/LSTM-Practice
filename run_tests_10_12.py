import attempted_lstm_9_18 as pwr_ball_pred
import time

# 1/69 = 1.449275362318841

#res_1 = pwr_ball_pred.main(69, 69, 6, .0001, 1, 1)
#2.6345933562428407
this_list = []

#layer_one_dep, layer_two_dep, layer_three_dep, sample_len, learn_rate, epoch_num, batch_num

res_1 = pwr_ball_pred.main(345,345,20,.00001,5,1024)
print(res_1)
this_list.append(res_1)
time.sleep((10*60))

res_2 = pwr_ball_pred.main(69,69,50,.0000001,50,1024)
print(res_2)
this_list.append(res_2)
time.sleep((10*60))

res_3 = pwr_ball_pred.main(345,69,24,.0000001,1000,1024)
print(res_3)
this_list.append(res_3)
time.sleep((10*60))

res_4 = pwr_ball_pred.main(276,276,6,.0001,1,1)
print(res_4)
this_list.append(res_4)
time.sleep((10*60))

res_5 = pwr_ball_pred.main(276,276,18,.000001,100,1024)
print(res_5)
this_list.append(res_5)
time.sleep((10*60))

res_6 = pwr_ball_pred.main(207,276,12,.0000001,50,512)
print(res_6)
this_list.append(res_6)
time.sleep((10*60))

res_7 = pwr_ball_pred.main(276,345,36,.0000005,20,512)
print(res_7)
this_list.append(res_7)
time.sleep((10*60))

res_8 = pwr_ball_pred.main(207,207,12,.00005,2,1024)
print(res_8)
this_list.append(res_8)
print(*this_list)