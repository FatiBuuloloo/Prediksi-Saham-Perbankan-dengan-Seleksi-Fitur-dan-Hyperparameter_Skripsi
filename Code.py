import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from datetime import datetime
import pandas_ta as ta
import random
import tensorflow as tf
import os
import pickle
import string
import re
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout,Input
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error, mean_absolute_percentage_error

def clean_data(datasets):
    datasets.columns = datasets.columns.str.strip()
    datasets = datasets.drop(columns=["Volume", "Adj Close"], errors='ignore')
    datasets["Open"] = datasets["Open"].astype(str)
    datasets = datasets[~datasets["Open"].str.contains("Dividend", na=False)]
    datasets = datasets[~datasets["Open"].str.contains("Stock Splits", na=False)]
    datasets["Open"] = datasets["Open"].str.replace(",", "", regex=True).astype(float
    cols_to_convert = ["Open", "High", "Low", "Close"]
    for col in cols_to_convert:
        if col in datasets.columns:
            datasets[col] = datasets[col].astype(str).str.replace(",", "", regex=True).astype(float)
            datasets[col] = np.round(datasets[col])
    datasets = datasets.dropna()
    datasets = datasets[::-1].reset_index(drop=True)

    return datasets
class CompanyAwareNormalizer:
    def __init__(self):
        self.scalers_all = {}
        self.scalers_col = {}  
    
    def fit_transform_all(self, nama_perusahaan, data):
        if nama_perusahaan not in self.scalers_all:
            self.scalers_all[nama_perusahaan] = MinMaxScaler()
        
        scaler = self.scalers_all[nama_perusahaan]
        return scaler.fit_transform(data)
    
    def inverse_transform_all(self, nama_perusahaan, scaled_data):
        if nama_perusahaan not in self.scalers_all:
            raise ValueError(f"Scaler untuk perusahaan {nama_perusahaan} tidak ditemukan")
        
        return self.scalers_all[nama_perusahaan].inverse_transform(scaled_data)
    
    def fit_transform_column(self, nama_perusahaan, column, data):
        if nama_perusahaan not in self.scalers_col:
            self.scalers_col[nama_perusahaan] = {}
            
        if column not in self.scalers_col[nama_perusahaan]:
            self.scalers_col[nama_perusahaan][column] = MinMaxScaler()
        
        scaler = self.scalers_col[nama_perusahaan][column]
        return scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    
    def inverse_transform_column(self, nama_perusahaan, column, scaled_data):
        if nama_perusahaan not in self.scalers_col:
            raise ValueError(f"Scaler untuk perusahaan {nama_perusahaan} tidak ditemukan")
            
        if column not in self.scalers_col[nama_perusahaan]:
            raise ValueError(f"Kolom {column} tidak dinormalisasi untuk perusahaan {nama_perusahaan}")
        
        scaler = self.scalers_col[nama_perusahaan][column]
        return scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()
def indikator(data):
    data = data.copy()
    data[["Open", "High", "Low", "Close"]] = data[["Open", "High", "Low", "Close"]].replace(",", "", regex=True
    rsi = ta.rsi(data["Close"])
    macd = ta.macd(data["Close"]).iloc[:, [0, 1]]
    dpo = ta.dpo(data['Close'])
    bias = ta.bias(data['Close'])
    bb = ta.bbands(data["Close"]).iloc[:, [1, 3, 4]]
    roc = ta.roc(data["Close"])
    mom = ta.mom(data["Close"])
    stoch_rsi = ta.stochrsi(data["Close"])
    rvi = ta.rvi(data["Close"])
    data_gabung = pd.concat([data, rsi, macd, dpo, bias, bb, roc, mom, stoch_rsi, rvi], axis=1)
    data_gabung = data_gabung.dropna()
    return data_gabung

def create_individual():
    neurons1 = random.randint(10, 200)
    neurons2 = random.randint(10, 200)
    dropout = round(random.uniform(0.1, 0.5), 2)
    epochs = random.randint(20, 200)
    
    feature_mask  = [0] * n_ohlc_indicators
    for idx in locked_feature:
        feature_mask [idx] = 1
    for group in grup_fitur:
        group_decision = random.randint(0,1)
        for idx in group:
            feature_mask [idx] = group_decision
    fitur_tanpa_grup = [i for i in range(n_ohlc_indicators) 
                          if i not in locked_feature 
                          and not any(i in grp for grp in grup_fitur)]
    for i in fitur_tanpa_grup:
        feature_mask [i] = random.randint(0,1)
    while sum(feature_mask ) == sum([feature_mask [i] for i in locked_feature]):
        for i in fitur_tanpa_grup:
            feature_mask [i] = random.randint(0,1)    
    return [neurons1, neurons2, dropout, epochs, feature_mask ]

def evaluate_individual(individual, gen_num, ind_num):
    try:
        neurons1, neurons2, dropout, epochs, feature_mask = individual
        teknikal_terpilih = np.where(feature_mask)[0]
        
        if len(teknikal_terpilih) == 0:
            print("Tidak ada fitur terpilih! Skip individu ini.")
            return None, None, float('inf')
            
        fitur_terpilih = list(teknikal_terpilih) + company_features
        
        model = Sequential()
        model.add(LSTM(neurons1, return_sequences=True, 
                    input_shape=(timesteps, len(fitur_terpilih))))
        model.add(LSTM(neurons2))
        model.add(Dropout(dropout))
        model.add(Dense(1))        
        model.compile(optimizer='adam', loss='logcosh')        
        early_stop = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)        
        history = model.fit(
            X_train[:, :, fitur_terpilih], y_train,
            epochs=epochs,
            validation_data=(X_val[:, :, fitur_terpilih], y_val),
            verbose=0,
            callbacks=[early_stop]
        )
        
        # Prediksi
        test_pred = [model.predict(X_test_all[i][:, :, fitur_terpilih]).flatten() 
                     for i in range(len(X_test_all))]
        
        data_asli = []
        data_prediksi = []
        
        for i in range(len(perusahaan)):
            try:
                # Perhatikan parameter yang benar untuk inverse_transform_column
                actual = normalizer.inverse_transform_column(
                    perusahaan[i], 
                    "Close",
                    scaled_data=y_test_all[i]
                )
                pred = normalizer.inverse_transform_column(
                    perusahaan[i],
                    "Close",
                    scaled_data=test_pred[i]
                )
                
                min_length = min(len(actual), len(pred))
                data_asli.append(actual[-min_length:])
                data_prediksi.append(pred[-min_length:])
            except Exception as e:
                print(f"Error denormalisasi perusahaan {perusahaan[i]}: {str(e)}")
                return None, None, float('inf')
        
        # Hitung metrik
        rmse = []
        mape = []
        
        for i in range(len(perusahaan)):
            try:
                rmse_i = np.sqrt(mean_squared_error(data_asli[i], data_prediksi[i]))
                mape_i = np.mean(np.abs((data_asli[i] - data_prediksi[i]) / data_asli[i])) * 100
                rmse.append(rmse_i)
                mape.append(mape_i)
            except Exception as e:
                print(f"Error hitung metrik {perusahaan[i]}: {str(e)}")
                rmse.append(float('inf'))
                mape.append(float('inf'))
        
        # Plotting
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')        
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.7f'))  # 7 desimal
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(7))  # Jumlah tick
        plt.title(f'Gen {gen_num+1} - Ind {ind_num+1}\nNeurons: {neurons1}/{neurons2} | Dropout: {dropout} | Epochs: {len(history.epoch)}/{epochs}')
        plt.legend()
        plt.subplot(1, 2, 2)
        tech_features = '\n'.join([f'- {feature_names[i]}' for i in teknikal_terpilih]) if len(teknikal_terpilih) > 0 else '- None'
        
        feature_group_mapping = {
            'MACD_12_26_9': 'MACD',
            'MACDh_12_26_9': 'MACD',
            'BBM_5_2.0': 'BB',
            'BBB_5_2.0': 'BB', 
            'BBP_5_2.0': 'BB',
            'STOCHRSIk_14_14_3_3': 'Stochastic RSI',
            'STOCHRSId_14_14_3_3': 'Stochastic RSI'
        }
        
        # Ambil nama grup unik dari fitur terpilih
        selected_features = [feature_names[i] for i in teknikal_terpilih]
        groups = []
        for feature in selected_features:
            group = feature_group_mapping.get(feature, feature)
            groups.append(group)
        unique_groups = sorted(list(set(groups)))
        tech_features = '\n'.join([f'- {group}' for group in unique_groups]) if unique_groups else '- None'
        
        text_content = [
            f"Generasi: {gen_num+1}",
            f"Individu: {ind_num+1}",
            "\nIndikator Teknikal:",
            tech_features,
            "\nPerusahaan:",
            ', '.join(company_names)
        ]
        plt.text(0.05, 0.85, '\n'.join(text_content), 
                 ha='left', va='top', 
                 fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFIK_DIR, f"Gen{gen_num+1}_Ind{ind_num+1}_loss.png"), dpi=300)
        plt.close()
        print(f"""
        === EVALUASI GEN {gen_num+1} - IND {ind_num+1} ===
        Neuron 1: {neurons1} | Neuron 2: {neurons2}
        Dropout: {dropout} | Epochs: {len(history.epoch)}/{epochs}
        Val Loss: {min(history.history['val_loss']):.7f}
        Indikator Terpilih: {', '.join(unique_groups) if unique_groups else 'Tidak ada'}
        ----------------------------------------
        BBCA:  RMSE: {rmse[0]:.4f} | MAPE: {mape[0]:.4f}%
        BBNI:  RMSE: {rmse[1]:.4f} | MAPE: {mape[1]:.4f}%
        BBRI:  RMSE: {rmse[2]:.4f} | MAPE: {mape[2]:.4f}%
        BBTN:  RMSE: {rmse[3]:.4f} | MAPE: {mape[3]:.4f}%
        BMRI:  RMSE: {rmse[4]:.4f} | MAPE: {mape[4]:.4f}%
        """)
        
        return model, history.history, min(history.history['val_loss']),rmse, mape
        
    except Exception as e:
        print(f"Error pada evaluasi: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, float('inf')
        
def mutate(individual, mutation_rate=0.2):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            if i == 4:
                # Mutasi per kelompok
                for group in grup_fitur:
                    if random.random() < mutation_rate:
                        current_value = mutated[i][group[0]]
                        new_value = 1 - current_value
                        for idx in group:
                            mutated[i][idx] = new_value
                
                # Mutasi fitur individual
                fitur_tanpa_grup = [i for i in range(n_ohlc_indicators)
                                      if i not in locked_feature 
                                      and not any(i in grp for grp in grup_fitur)]
                for idx in fitur_tanpa_grup:
                    if random.random() < mutation_rate:
                        mutated[i][idx] = 1 - mutated[i][idx]
            else:
                # Mutasi parameter lain
                if i == 0 or i == 1:
                    mutated[i] = random.randint(10, 200)
                elif i == 2:
                    mutated[i] = round(random.uniform(0.1, 0.5), 2)
                elif i == 3:
                    mutated[i] = random.randint(20, 100)
    return mutated

def crossover(parent1, parent2):
    child = parent1.copy()    
    # Crossover parameter
    for i in [0,1,2,3]:
        if random.random() < 0.5:
            child[i] = parent2[i]    
    # Crossover feature mask
    child_mask = parent1[4].copy()    
    # Kelompok fitur
    for group in grup_fitur:
        if random.random() < 0.5:
            for idx in group:
                child_mask[idx] = parent2[4][idx]    
    # Fitur individual
    fitur_tanpa_grup = [i for i in range(n_ohlc_indicators)
                          if i not in locked_feature 
                          and not any(i in grp for grp in grup_fitur)]
    for idx in fitur_tanpa_grup:
        if random.random() < 0.5:
            child_mask[idx] = parent2[4][idx]    
    child[4] = child_mask    
    return child
def save_checkpoint(current_gen, current_ind_in_gen, population, best_individual, best_fitness, best_history, best_model, best_gen, best_ind, list_loss, list_rmse, list_mape):
    best_history = best_history if best_history is not None else {'loss': [], 'val_loss': []}
    best_model = best_model if best_model is not None else create_individual()
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    model_filename = f"best_model_gen{current_gen+1}_ind{current_ind_in_gen+1}.h5"
    model_path = os.path.join(BEST_MODEL_DIR, model_filename)
    
    try:
        best_model.save(model_path)
    except Exception as e:
        print(f"Gagal menyimpan model: {str(e)}")
        return
    checkpoint = {
        'generasi': current_gen,
        'individu': current_ind_in_gen,
        'populasi': population,
        'individu_terbaik': best_individual,
        'fitness_terbaik': best_fitness,
        'loss_terbaik': best_history,
        'lokasi_model_terbaik': model_path,
        'generasi_terbaik' : best_gen,
        'individu_terbaik' : best_ind,
        'kumpulan_loss' : list_loss,
        'kumpulan_rmse' : list_rmse,
        'kumpulan_mape' : list_mape, 
        'config': { 
            'population_size': population_size,
            'generations': generations,
            'feature_names': feature_names,
            'company_names': company_names
        }
    }
    
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: Gen {current_gen+1} Ind {current_ind_in_gen+1}")


def load_checkpoint():
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        next_gen = checkpoint['generasi']
        next_ind = checkpoint['individu'] + 1 
        last_best_gen = checkpoint['generasi_terbaik']
        last_best_ind = checkpoint['individu_terbaik']
        last_list_loss = checkpoint['kumpulan_loss']
        last_list_rmse  = checkpoint['kumpulan_rmse']
        last_list_mape  = checkpoint['kumpulan_mape']
        if next_ind >= population_size:
            next_gen += 1
            next_ind = 0
            
        if next_gen >= generations:
            print("Training sudah selesai")
            return None
            
        print(f"Melanjutkan dari Generasi {next_gen+1} Individu {next_ind+1}")
        return {
            'generasi': next_gen,
            'individu': next_ind,
            'populasi': checkpoint['populasi'],
            'individu_terbaik': checkpoint['individu_terbaik'],
            'fitness_terbaik': checkpoint['fitness_terbaik'],
            'loss_terbaik': checkpoint['loss_terbaik'],
            'best_model': tf.keras.models.load_model(checkpoint['lokasi_model_terbaik']),
            'generasi_terbaik' : last_best_gen,
            'individu_terbaik' : last_best_ind,
            'kumpulan_loss' : last_list_loss,
            'kumpulan_rmse' : last_list_rmse,
            'kumpulan_mape' : last_list_mape
        }
    except FileNotFoundError:
        print("Tidak ditemukan Checkpoint, dimulai dari awal")
        return None
    except Exception as e:
        print(f"Gagal Memuat checkpoint: {str(e)}")
        return None
def run_training():
    checkpoint_data = load_checkpoint()
    if checkpoint_data:
        start_gen = checkpoint_data['generasi']
        start_ind_in_gen = checkpoint_data['individu']
        population = checkpoint_data['populasi']
        best_individual = checkpoint_data['individu_terbaik']
        best_fitness = checkpoint_data['fitness_terbaik']
        best_history = checkpoint_data['loss_terbaik']
        best_model = checkpoint_data['best_model']
        best_gen = checkpoint_data['generasi_terbaik']
        best_ind = checkpoint_data['individu_terbaik']
        list_loss = checkpoint_data['kumpulan_loss']
        list_rmse = checkpoint_data['kumpulan_rmse']
        list_mape = checkpoint_data['kumpulan_mape']
    else:
        start_gen = 0
        start_ind_in_gen = 0
        population = [create_individual() for _ in range(population_size)]
        best_fitness = float('inf')
        best_individual = None
        best_history = {'loss': [], 'val_loss': []}
        best_model = None
        best_gen = []
        best_ind = []
        list_loss = {}
        list_rmse = {}
        list_mape = {}
        for perusahaan_name in perusahaan:
            list_rmse[perusahaan_name] = []
            list_mape[perusahaan_name] = []
    try:
        for gen in range(start_gen, generations):
            print(f"""
            Generasi {gen+1}/{generations}
            """)
            
            # Cek apakah generasi ini perlu diproses
            if gen > start_gen:
                start_ind_in_gen = 0
            fitness=[]
            fitness_dict = list_loss
            best_gen_now = best_gen
            best_ind_now = best_ind
            rmse_dict = list_rmse 
            mape_dict = list_mape
            for idx_in_gen in range(start_ind_in_gen, population_size):
                individual = population[idx_in_gen]
                
                try:
                    model, history, fit, rmse1, mape1 = evaluate_individual(individual, gen, idx_in_gen)
                    fitness.append(fit)
                    fitness_dict[(gen+1, idx_in_gen+1)] = fit
                    for company, data1 in zip(perusahaan, rmse1):
                        rmse_dict [company].append(data1)
                    for company, data2 in zip(perusahaan, mape1):
                        mape_dict [company].append((data2))
                    if fit < best_fitness and model:
                        best_fitness = fit
                        best_model = model
                        best_history = history
                        best_individual = individual
                        print(f"\nBEST UPDATE: Gen {gen+1} Ind {idx_in_gen+1} | Loss: {best_fitness:.7f}")
                        best_gen.append(gen+1)
                        best_ind.append(idx_in_gen+1)
                    save_checkpoint(
                        gen,
                        idx_in_gen, 
                        population,
                        best_individual,
                        best_fitness,
                        best_history,
                        best_model,
                        best_gen_now,
                        best_ind_now,
                        fitness_dict,
                        rmse_dict,
                        mape_dict
                    )
                    
                except Exception as e:
                    print(f"Error di Gen {gen+1} Ind {idx_in_gen+1}: {str(e)}")
                    continue
            sorted_pop = [x for _, x in sorted(zip(fitness, population))]
            elite = sorted_pop[:max(1, int(population_size*0.2))]
            
            new_pop = elite.copy()
            while len(new_pop) < population_size:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                child = mutate(crossover(parent1, parent2))
                new_pop.append(child)
            
            population = new_pop
            
            print(f"Generation {gen+1} completed ")
            print(f"Nilai Val Loss Terendah berada pada Generasi : {best_gen[-1]} Individu : {best_ind[-1]} | Best Loss: {best_fitness:.7f}")
        print("\nTraining selesai!")
        
    except KeyboardInterrupt:
        sorted_val_items1 = sorted(fitness_dict.items(), key=lambda x: (x[0][0], x[0][1]))
        sorted_val_items = sorted(fitness_dict.items(), key=lambda x: (x[0][0], x[0][1]))        
        x = [f'G{gen}-I{ind}' for (gen, ind), val in sorted_val_items]
        y = [val for (gen_ind), val in sorted_val_items1]
        gen_counts = {}
        for (gen, _), val in sorted_val_items:
            gen_counts[gen] = gen_counts.get(gen, 0) + 1
        tick_positions = []
        tick_labels = []
        pos = 0
        for gen, count in gen_counts.items():
            tick_positions.append(pos)  
            tick_labels.append(f'Gen {gen}')
            pos += count
        plt.figure(figsize=(14, 5))
        plt.plot(x, y, marker='o', linestyle='-', color='blue')
        plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=0)
        plt.xlabel('Generasi')
        plt.ylabel('Validation Loss')
        plt.title('Perkembangan Validation Loss per Individu per Generasi')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        save_dict_to_csv(rmse_dict, f"RMSE_Model.csv")
        save_dict_to_csv(mape_dict, f"MAPE_Model.csv")
        print("\nTraining dijeda! Gunakan final_evaluation() untuk melihat hasil sementara")
save_path = r"E:\Kuliah\Skripsi\Model LSTM-GA Skripsi\Metrik Evaluasi"
os.makedirs(save_path, exist_ok=True)
def save_dict_to_csv(data_dict, filename):
    df = pd.DataFrame(dict(sorted(data_dict.items())))
    full_path = os.path.join(save_path, filename)
    df.to_csv(full_path, index=False)
    print(f"File '{filename}' berhasil disimpan ke: {full_path}")
def final_evaluation(model, gen_num1, ind_num1,fitur1):
    feature_group_mapping = {
        'BB': [6, 7, 8],                   # BBM, BBB, BBP
        'Stochastic RSI': [11, 12],         # STOCHRSIk dan STOCHRSId
        'MACD': [2, 3]                      # MACD dan MACDh
    }
    teknikal_terpilih = []
    for feature in fitur1:
        if feature in feature_group_mapping:
            teknikal_terpilih.extend(feature_group_mapping[feature])
        else:
            teknikal_terpilih.append(feature_names.index(feature))
    fitur_terpilih = list(set(teknikal_terpilih)) + company_features
    hasil_prediksi = [model.predict(X_test_all[i][:, :, fitur_terpilih]).flatten() 
                     for i in range(len(perusahaan))]
    data_asli = []
    data_prediksi = []
    for i in range(len(perusahaan)):
        actual = normalizer.inverse_transform_column(
            perusahaan[i],
            "Close",
            y_test_all[i]
        )
        pred = normalizer.inverse_transform_column(
            perusahaan[i],
            "Close",
            hasil_prediksi[i]
        )
        min_len = min(len(actual), len(pred))
        data_asli.append(actual[-min_len:])
        data_prediksi.append(pred[-min_len:])
    rmse1 = [np.sqrt(mean_squared_error(a, p)) for a, p in zip(data_asli, data_prediksi)]
    mape1 = [np.mean(np.abs((a - p)/a)) * 100 for a, p in zip(data_asli, data_prediksi)]
    
    print(f"""
    === EVALUASI GEN {gen_num1} - IND {ind_num1} ===
    ----------------------------------------
    BBCA:  RMSE: {rmse1[0]:.4f} | MAPE: {mape1[0]:.4f}%
    BBNI:  RMSE: {rmse1[1]:.4f} | MAPE: {mape1[1]:.4f}%
    BBRI:  RMSE: {rmse1[2]:.4f} | MAPE: {mape1[2]:.4f}%
    BBTN:  RMSE: {rmse1[3]:.4f} | MAPE: {mape1[3]:.4f}%
    BMRI:  RMSE: {rmse1[4]:.4f} | MAPE: {mape1[4]:.4f}%
    """)
filenames = ["Bank/BBCA.csv", "Bank/BBNI.csv", "Bank/BBRI.csv", "Bank/BBTN.csv", "Bank/BMRI.csv"]
raw_data = [pd.read_csv(file) for file in filenames]
perusahaan = [f.split("/")[-1].split(".")[0] for f in filenames]
for df in raw_data:
    df.columns = df.columns.str.strip()
cleaned_data = [clean_data(i) for i in raw_data]
data_indikator = [indikator(data) for data in cleaned_data]
list_tanggal = [data_indikator[i].Date for i in range (len(perusahaan))]
list_tanggal = [np.array(list_tanggal[i]).reshape(-1, 1) for i in range (len(perusahaan))]
from datetime import datetime

tanggal_unproses_perusahaan = []
tanggal_unproses_training = []
tanggal_unproses_validasi = []
tanggal_unproses_tes = []

SEQ_LENGTH = 14
n_future = 1

for j in range(len(perusahaan)):  
    tanggal_unproses = []
    
    for i in range(SEQ_LENGTH, len(data_indikator[j]) - n_future + 1):
        tanggal_str = list_tanggal[j][i + n_future - 1, 0]
        if isinstance(tanggal_str, str):  
            tanggal_date = datetime.strptime(tanggal_str, "%d-%b-%y").date()  
        else:
            tanggal_date = tanggal_str 
        tanggal_unproses.append(tanggal_date)  
    n = len(tanggal_unproses)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)    

    tanggal_unproses_perusahaan.append(tanggal_unproses)  
    tanggal_unproses_training.append(tanggal_unproses[:train_end])
    tanggal_unproses_validasi.append(tanggal_unproses[train_end:val_end])
    tanggal_unproses_tes.append(tanggal_unproses[val_end:])
for j, nama_perusahaan in enumerate(perusahaan):
    print(f"Perusahaan: {nama_perusahaan}, Tanggal Target: {len(tanggal_unproses_perusahaan[j])}, Training : {len(tanggal_unproses_training[j])}, Validasi : {len(tanggal_unproses_validasi[j])}, Tes : {len(tanggal_unproses_tes[j])}")  
data_indikator = [data_indikator[i].drop(columns=["Date", "Open", "High", "Low"]) for i in range(len(data_indikator))]
for i in range(len(cleaned_data)):
    cleaned_data[i] = cleaned_data[i].iloc[:,1:].apply(lambda x: pd.to_numeric(x, errors="coerce") if x.dtype == "object" else x)
for idx, df in enumerate(data_indikator):
    correlation = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar=True)
    plt.title(f'Correlation Heatmap - {perusahaan[idx]}')
    plt.tight_layout()
    filename = f'heatmap_{perusahaan[idx]}.png'
    plt.savefig(os.path.join(output_path_graph, filename))
    plt.close()
print("Semua heatmap korelasi telah berhasil disimpan.")

column = ["Close"]
normalizer = CompanyAwareNormalizer()
scaled_data_all = {}
scaled_data_close = {}
for company, data in zip(perusahaan, data_indikator):
    scaled_all = normalizer.fit_transform_all(company, data)
    scaled_data_all[company] = scaled_all 
    scaled_data_close[company] = normalizer.fit_transform_column(company, "Close", data["Close"])
onehot_encoder = OneHotEncoder(sparse=False)
perusahaan_ids = np.array(perusahaan).reshape(-1, 1)
onehot_encoder.fit(perusahaan_ids)
encoded_perusahaan = onehot_encoder.transform(perusahaan_ids)
print(encoded_perusahaan)
scaled_data = {}
if len(perusahaan)==len(scaled_data_all):   
    for i in range (len(perusahaan)): 
        scaled_data[perusahaan[i]] = scaled_data_all[perusahaan[i]]
else :
    print("Jumlah Perusahaan tidak sama dengan jumlah data indikator")

X_train_all, X_val_all, X_test_all = [], [], []
y_train_all, y_val_all, y_test_all = [], [], []

SEQ_LENGTH = 14
n_future = 1
for company, data in scaled_data.items():
    X, y= [], []
    for i in range(SEQ_LENGTH , len(data) - n_future +1):
        X.append(data[i - SEQ_LENGTH:i, 0:data.shape[1]])
        y.append(data[i + n_future - 1:i + n_future, 0])
    X = np.array(X)
    y = np.array(y)
    company_id = onehot_encoder.transform([[company]])
    company_id_repeated = np.repeat(company_id, SEQ_LENGTH, axis=0)    
    X_combined = []
    for seq in X:
        seq_with_id = np.hstack([seq, company_id_repeated])
        X_combined.append(seq_with_id)
    
    X_combined = np.array(X_combined)
    n = len(X_combined)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)
    X_train = X_combined[:train_end]
    X_val = X_combined[train_end:val_end]
    X_test = X_combined[val_end:]    
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    X_train_all.append(X_train)
    X_val_all.append(X_val)
    X_test_all.append(X_test)
    
    y_train_all.append(y_train), y_val_all.append(y_val), y_test_all.append(y_test)
X_train = np.concatenate(X_train_all, axis=0)
X_val = np.concatenate(X_val_all, axis=0)
X_test = np.concatenate(X_test_all, axis=0)
y_train, y_val, y_test = np.concatenate(y_train_all, axis=0), np.concatenate(y_val_all, axis=0), np.concatenate(y_test_all, axis=0)

print(f"Training shape: X = {X_train.shape}, y_Close = {y_train.shape}")
print(f"Validation shape: X = {X_val.shape}, y_Close = {y_val.shape}")
print(f"Test shape: X = {X_test.shape}, y_Close = {y_test.shape}")

GRAFIK_DIR = "E:/Kuliah/Skripsi/Model LSTM-GA Skripsi/Grafik"                   
BEST_MODEL_DIR = "E:/Kuliah/Skripsi/Model LSTM-GA Skripsi/Best_Model"         
os.makedirs(GRAFIK_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
BEST_MODEL_PREFIX = "best_model"  

population_size = 20
generations = 50

CHECKPOINT_FILE = f"Model_GA_Skripsi.pkl"
SAVE_INTERVAL = 3
n_features = X_train.shape[2]
timesteps = X_train.shape[1]
locked_feature = [0]  # Indeks kolom Close 
grup_fitur = [
    [2, 3],           # MACD dan MACDh
    [6, 7, 8],        # BBM, BBB, BBP
    [11, 12]          # STOCHRSIk dan STOCHRSId
]
feature_names = [
    'Close', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'DPO_20',
    'BIAS_SMA_26', 'BBM_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0', 'ROC_10',
    'MOM_10', 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3', 'RVI_14'
]
company_names = ['BBCA', 'BBNI', 'BBRI', 'BBTN', 'BMRI']
n_ohlc_indicators = n_features - 5
company_features = list(range(n_ohlc_indicators, n_features))

checkpoint_data = load_checkpoint()
if checkpoint_data:
    start_gen = checkpoint_data['generasi']
    start_ind_in_gen = checkpoint_data['individu']
    population = checkpoint_data['populasi']
    best_individual = checkpoint_data['individu_terbaik']
    best_fitness = checkpoint_data['fitness_terbaik']
    best_history = checkpoint_data['loss_terbaik']
    best_model = checkpoint_data['best_model']
else:
    start_gen = 0
    start_ind_in_gen = 0
    population = [create_individual() for _ in range(population_size)]
    best_fitness = float('inf')
    best_individual = None
    best_history = {'loss': [], 'val_loss': []}
    best_model = None

run_training()

model_GA_akhir = tf.keras.models.load_model('E:/Kuliah/Skripsi/Model LSTM-GA SkripsiBest_Model/best_model_gen43_ind3.h5')

nama_file_model = 'best_model_gen2_ind2.h5'
match = re.search(r"gen(\d+)_ind(\d+)", nama_file_model)
if match:
    generasi_terbaik1 = int(match.group(1))
    individu_terbaik1 = int(match.group(2))
selected_features1 = [
        'Close', 'BIAS_SMA_26', 'MOM_10', 'Stochastic RSI'
    ]
final_evaluation(
    model_GA_akhir,
    generasi_terbaik1,
    individu_terbaik1,
    selected_features1
)
