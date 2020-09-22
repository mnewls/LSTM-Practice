import attempted_lstm_9_18 as pwr_ball_pred

#def main(layer_one_dep, layer_two_dep, sample_len, learn_rate, epoch_num, batch_num):

'''models_tried = {
    "layer_one_depth" : 69,
    "layer_two_depth" : 69,
    "sample length" : 6,
    "learning_rate" : .0001,
    "number_of_epoch" : 1,s
    "number_of_batches" : 1,
    "resulting_accuracy_percent" : 2.6345933562428407
}'''

# 1/69 = 1.449275362318841
#res_1 = pwr_ball_pred.main(69, 69, 6, .0001, 1, 1)
#2.6345933562428407
#res_2 = pwr_ball_pred.main(69, 69, 12, .0002, 2, 2)
#2.422145328719723
# res = pwr_ball_pred.main(69, 69, 24, .0001, 1, 1)
#1.5204678362573099

res = pwr_ball_pred.main(69, 69, 3, .001, 2, 2)
print(res)