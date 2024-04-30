import os


def getFromData(data_name, task_name):
    if task_name == "long_term_forecast":
        return getFromDataLongForecasting(data_name)
    elif task_name == "short_term_forecast":
        return getFromDataShortForecasting(data_name)
    elif task_name == "anomaly_detection":
        return getFromDataAnomalyDetection(data_name)
    elif task_name == "classification":
        return getFromDataClassification(data_name)
    elif task_name == "imputation":
        return getFromDataImputation(data_name)
    else:
        raise ValueError("Task name not recognized")


def getFromDataImputation(data_name):
    data, root_path, dim, _ = getFromDataLongForecasting(data_name)
    mask_rate = [0.125, 0.25, 0.375, 0.5]
    return data, root_path, dim, mask_rate


def getFromDataClassification(data_name):
    data = data_name
    root_path = './dataset/' + data_name + '/'
    return data, root_path, None, [0]


def getFromDataAnomalyDetection(data_name):
    data = data_name
    root_path = './dataset/' + data_name
    if data_name == 'MSL':
        dim = 55
    elif data_name in ['PSM', 'SMAP']:
        dim = 25
    elif data_name == 'SMD':
        dim = 38
    elif data_name == 'SWAT':
        dim = 51
    pred_lens = [0]
    return data, root_path, dim, pred_lens


def getFromDataShortForecasting(data_name):
    data = data_name
    root_path = './dataset/m4'
    dim = 1
    pred_lens = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    # pred_lens = ['Weekly']
    return data, root_path, dim, pred_lens


def getFromDataLongForecasting(data_name):
    pred_lens = [96, 192, 336, 720]

    if data_name in ['ETTh2', 'ETTh1', 'ETTm1', 'ETTm2', 'short', 'long']:
        data = data_name
        root_path = './dataset/ETT-small/'
        dim = 7
        pred_lens = [192]
    else:
        data = 'custom'
        if data_name == 'weather':
            root_path = './dataset/weather/'
            dim = 21  # weather
            # pred_lens = [720]
        elif data_name == 'exchange_rate':
            root_path = './dataset/exchange_rate/'
            dim = 8  # exchange_rate
        elif data_name == 'electricity':
            root_path = './dataset/electricity/'
            dim = 321  # ECL
            pred_lens = [336, 720]
            # pred_lens = [192, 336, 720]
        elif data_name == 'traffic':
            root_path = './dataset/traffic/'
            dim = 862
            pred_lens = [192, 336, 720]
        elif data_name in ['national_illness', 'a']:
            root_path = './dataset/illness/'
            dim = 7
            pred_lens = [24, 36, 48, 60]

    return data, root_path, dim, pred_lens


def run_long_term_forecasting(model, data, data_name, pred_len, root_path, data_path, dim, d_model, d_ff, train_epochs):
    # Monarch
    task = 'long_term_forecast'
    batch_size = 4
    if model == 'Crossformer':
        if data_name == 'electricity':
            d_model = 256
            d_ff = 512
        elif data_name == 'weather':
            d_model = 32
            d_ff = 32
    model_id = "{}_{}_{}_{}".format(model, data_name, pred_len, 'mo')
    scripts = "python run.py --task_name {} --is_training 0 --root_path {} --data_path {} --model_id {} --patience 1\
                                      --model {} --data {} --features M --seq_len 96 --label_len 48 --pred_len {} --e_layers 2 --d_layers 1 \
                                      --factor 3 --enc_in {} --dec_in {} --c_out {} --des 'Exp' --d_model {} --d_ff {} --itr 1 --batch_size {} --use_monarch".format(
        task, root_path, data_path, model_id, model,
        data, pred_len, dim, dim, dim, d_model, d_ff, batch_size)
    if model == 'Crossformer':
        if data_name == 'electricity':
            scripts += " --n_heads 2"
    if model == 'Pyraformer':
        scripts += " --train_epochs 3"#1069 1135
    os.system(scripts)
    #  --use_monarch
    # No Monarch
    model_id = "{}_{}_{}".format(model, data_name, pred_len)
    scripts = "python run.py --task_name {} --is_training 0 --root_path {} --data_path {} --model_id {} --patience 1\
                                          --model {} --data {} --features M --seq_len 96 --label_len 48 --pred_len {} --e_layers 2 --d_layers 1 \
                                          --factor 3 --enc_in {} --dec_in {} --c_out {} --des 'Exp' --d_model {} --d_ff {} --itr 1 --batch_size {}".format(
        task, root_path, data_path, model_id, model,
        data, pred_len, dim, dim, dim, d_model, d_ff, batch_size)
    if model == 'Crossformer':
        if data_name == 'electricity':
            scripts += " --n_heads 2"
    if model == 'Pyraformer':
        scripts += " --train_epochs 3"
    os.system(scripts)


def run_short_term_forecasting(model, data, data_name, pred_len, root_path, dim, d_model, d_ff):
    task = 'short_term_forecast'

    # Monarch
    model_id = "{}_{}_{}_{}".format(model, data_name, pred_len, 'mo')
    script = "python run.py --task_name {} --is_training 1 --root_path {} --model_id {} \
                                          --model {} --data {} --features M --seasonal_patterns {} --e_layers 2 --d_layers 1 \
                                          --factor 3 --enc_in {} --dec_in {} --c_out {} --batch_size 64 --des 'Exp' \
                                          --d_model {} --d_ff {} --itr 1 --learning_rate 0.001 --loss SMAPE --use_monarch".format(
        task, root_path, model_id, model,
        data, pred_len, dim, dim, dim, d_model, d_ff)
    if model in ['Nonstationary_Transformer', 'PatchTST']:
        script += " --p_hidden_dims 256 256 --p_hidden_layers 2"
    # os.system(script)

    # No Monarch
    model_id = "{}_{}_{}".format(model, data_name, pred_len)
    script = "python run.py --task_name {} --is_training 1 --root_path {} --model_id {} \
                                              --model {} --data {} --features M --seasonal_patterns {} --e_layers 2 --d_layers 1 \
                                              --factor 3 --enc_in {} --dec_in {} --c_out {} --batch_size 16 --des 'Exp' \
                                              --d_model {} --d_ff {} --itr 1 --learning_rate 0.001 --loss SMAPE".format(
        task, root_path, model_id, model,
        data, pred_len, dim, dim, dim, d_model, d_ff)
    if model in ['Nonstationary_Transformer', 'PatchTST']:
        script += " --p_hidden_dims 256 256 --p_hidden_layers 2"
    os.system(script)


def run_anomaly_detection(model, data, root_path, dim):
    task = 'anomaly_detection'
    ratio = 1
    batch_size = 128
    if data == 'SMD':
        ratio = 0.5
    if model == 'PatchTST':
        batch_size = 32
    # Monarch
    model_id = "{}_{}_{}".format(model, data, 'mo')
    script = "python run.py --task_name {} --is_training 1 --root_path {} --model_id {} \
                                          --model {} --data {} --features M --seq_len 100 --pred_len 0 --e_layers 3 \
                                          --enc_in {} --c_out {} --anomaly_ratio {} --batch_size {} --des 'Exp' \
                                          --d_model 128 --d_ff 128 --itr 1 --use_monarch".format(
        task, root_path, model_id, model,
        data, dim, dim, ratio, batch_size)
    if data == 'SWAT':
        script += " --top_k 3"
    os.system(script)

    # No Monarch


def run_classification(model, data, root_path):
    task = 'classification'
    batch_size = 16  # 16
    learning_rate = 0.001  # 0.001
    if model == 'PatchTST':
        batch_size = 2
    # Monarch
    model_id = "{}_{}".format(data, 'mo')
    script = "python run.py --task_name {} --is_training 1 --root_path {} --model_id {} \
                                          --model {} --data {} --e_layers 3 --patience 10 \
                                          --batch_size {} --des 'Exp' --learning_rate {} --train_epochs 100 \
                                          --d_model 128 --d_ff 256 --top_k 3 --itr 1 --use_monarch".format(
        task, root_path, model_id, model,
        'UEA', batch_size, learning_rate)

    # model_id = "{}".format(data)
    # script = "python run.py --task_name {} --is_training 1 --root_path {} --model_id {} \
    #                                           --model {} --data {} --e_layers 3 --patience 10 \
    #                                           --batch_size {} --des 'Exp' --learning_rate 0.00001 --train_epochs 100 \
    #                                           --d_model 128 --d_ff 256 --top_k 3 --itr 1".format(
    #     task, root_path, model_id, model,
    #     'UEA', batch_size)
    if model == 'iTransformer':
        script += " --enc_in 3"
    os.system(script)

    # No Monarch


def run_imputation(model, data, data_name, root_path, data_path, dim, mask_rate):
    task = 'imputation'

    # Monarch
    model_id = "{}_mask_{}_{}".format(data_name, mask_rate, 'mo')
    script = "python run.py --task_name {} --is_training 1 --root_path {} --data_path {} --model_id {} --mask_rate {} \
                                          --model {} --data {} --features M --seq_len 96 --label_len 0 --pred_len 0 --e_layers 2 \
                                          --d_layers 1 --factor 3 --enc_in {} --enc_in {} --c_out {} --batch_size 16 --des 'Exp' \
                                          --d_model 128 --d_ff 128 --itr 1 --top_k 5 --learning_rate 0.001 --use_monarch".format(
        task, root_path, data_path, model_id, mask_rate, model,
        data, dim, dim, dim)
    if model == 'Nonstationary_Transformer':
        script += " --p_hidden_dims 256 256 --p_hidden_layers 2"
    # os.system(script)

    # No Monarch
    model_id = "{}_mask_{}".format(data_name, mask_rate)
    script = "python run.py --task_name {} --is_training 1 --root_path {} --data_path {} --model_id {} --mask_rate {} \
                                              --model {} --data {} --features M --seq_len 96 --label_len 0 --pred_len 0 --e_layers 2 \
                                              --d_layers 1 --factor 3 --enc_in {} --enc_in {} --c_out {} --batch_size 16 --des 'Exp' \
                                              --d_model 128 --d_ff 128 --itr 1 --top_k 5 --learning_rate 0.001".format(
        task, root_path, data_path, model_id, mask_rate, model,
        data, dim, dim, dim)
    if model == 'Nonstationary_Transformer':
        script += " --p_hidden_dims 256 256 --p_hidden_layers 2"
    os.system(script)


if __name__ == '__main__':
    # models = ['Transformer', 'Informer', 'Autoformer', 'FEDformer', 'Crossformer', 'Pyraformer',
    #           'Nonstationary_Transformer', 'iTransformer', 'PatchTST']
    # models = ['Pyraformer', 'Nonstationary_Transformer', 'iTransformer', 'PatchTST']
    models = ['Crossformer']
    # models = ['Nonstationary_Transformer']

    task = 'long_term_forecast'
    # task = 'short_term_forecast'
    # task = 'anomaly_detection'
    # task = 'classification'
    # task = 'imputation'
    # datas = ['ETTm2', 'electricity', 'exchange_rate', 'traffic', 'weather', 'national_illness']
    # datas = ['PEMS-SF']
    # datas = ['weather', 'national_illness']
    datas = ['traffic']
    # datas = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'national_illness', 'electricity', 'traffic']
    # datas = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'weather']
    # datas = ['m4']
    # datas = ['MSL', 'PSM', 'SMAP', 'SMD', 'SWAT']
    # datas = ['EthanolConcentration', 'FaceDetection', 'Handwriting', 'Heartbeat', 'JapaneseVowels', 'PEMS-SF',
    #          'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'UWaveGestureLibrary']
    # datas = ['electricity', 'traffic']
    # datas = ['ETTm1', 'ETTm2']
    # datas = ['national_illness', 'weather']
    # datas = ['long']

    for model in models:
        for data in datas:
            data_path = data + '.csv'
            data_name = data
            # if model in ['Autoformer', 'Crossformer']:
            #     train_epochs = 2
            # else:
            #     train_epochs = 10
            train_epochs = 10
            if model == "iTransformer":
                d_model = 128
                d_ff = 128
            else:
                d_model = 512
                d_ff = 2048

            data, root_path, dim, pred_lens = getFromData(data, task)
            for pred_len in pred_lens:
                if task == 'long_term_forecast':
                    run_long_term_forecasting(model, data, data_name, pred_len, root_path, data_path, dim, d_model,
                                              d_ff, train_epochs)
                elif task == 'short_term_forecast':
                    run_short_term_forecasting(model, data, data_name, pred_len, root_path, dim, d_model, d_ff)
                elif task == 'anomaly_detection':
                    run_anomaly_detection(model, data, root_path, dim)
                elif task == 'classification':
                    run_classification(model, data, root_path)
                elif task == 'imputation':
                    run_imputation(model, data, data_name, root_path, data_path, dim, pred_len)
