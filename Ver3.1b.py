import json
import random
from datetime import datetime
from collections import Counter
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Nama file untuk menyimpan dan memuat nomor
filename = 'numbers.json'

def load_numbers():
    """Memuat nomor dari file JSON."""
    try:
        with open(filename, 'r') as file:
            numbers = json.load(file)
            return numbers
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print("File JSON tidak valid.")
        return []

def save_numbers(numbers):
    """Menyimpan nomor ke file JSON."""
    with open(filename, 'w') as file:
        json.dump(numbers, file, indent=4)

def add_number(number, numbers):
    """Menambahkan nomor ke daftar jika belum ada."""
    if number not in numbers:
        numbers.append(number)
        save_numbers(numbers)
        print("Nomor berhasil ditambahkan.")
    else:
        print("Nomor sudah ada dalam daftar.")

def remove_number(number, numbers):
    """Menghapus nomor dari daftar jika ada."""
    if number in numbers:
        numbers.remove(number)
        save_numbers(numbers)
        print("Nomor berhasil dihapus.")
    else:
        print("Nomor tidak ditemukan dalam daftar.")

# Data historis nomor togel
numbers = load_numbers()

def predict_based_on_frequencies(numbers, num_predictions=5):
    """Prediksi nomor berdasarkan frekuensi kemunculan."""
    frequency = Counter(numbers)
    most_common = frequency.most_common(num_predictions)
    return [num for num, _ in most_common]

def prepare_data_for_ml(numbers):
    """Persiapkan data historis untuk pelatihan model machine learning."""
    if not numbers:
        raise ValueError("Daftar nomor kosong, tidak bisa menyiapkan data.")
    
    # Memastikan nomor adalah string 4 digit
    X = np.array([[int(num[i]) for i in range(4)] for num in numbers if len(num) == 4])
    y = np.array([int(num[-1]) for num in numbers if len(num) == 4])
    return X, y

def build_neural_network(input_shape):
    """Membangun model neural network menggunakan TensorFlow."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Output untuk 10 kelas (digit 0-9)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(X, y):
    """Latih model machine learning (Neural Network) untuk prediksi."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Membangun dan melatih model neural network
    model = build_neural_network(input_shape=X_train_scaled.shape[1])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

    # Evaluasi model
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
    print(f"Akurasi model: {test_acc:.2f}")

    return model, scaler

def predict_with_model(model, scaler, numbers, valid_last_digits):
    """Gunakan model untuk memprediksi nomor baru."""
    X_new = np.array([[int(num[i]) for i in range(4)] for num in numbers if len(num) == 4])
    X_new_scaled = scaler.transform(X_new)
    
    y_pred_probs = model.predict(X_new_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Prediksi digit terakhir

    predicted_numbers = [numbers[i] for i in range(len(y_pred)) if str(y_pred[i]) in valid_last_digits]
    return predicted_numbers

def generate_random_number_for_days(numbers, valid_last_digits):
    """Generate nomor acak berdasarkan digit dari posisi tertentu dan digit terakhir yang valid."""
    def get_random_digit(position):
        digits = [num[position] for num in numbers if len(num) == 4]
        return random.choice(digits)

    thousands = get_random_digit(0)
    hundreds = get_random_digit(1)
    tens = get_random_digit(2)
    units = random.choice(valid_last_digits)
    return thousands + hundreds + tens + units

def get_day_and_shio(number):
    """Dapatkan hari dan shio berdasarkan digit terakhir dari nomor."""
    units_digit = number[-1]
    days_mapping = {
        '0': 'Minggu', '1': 'Senin', '2': 'Selasa', '3': 'Rabu', '4': 'Kamis',
        '5': "Jum'at", '6': 'Sabtu'
    }
    shio_mapping = {
        '0': 'Monyet', '1': 'Naga', '2': 'Kelinci', '3': 'Harimau', '4': 'Kerbau', '5': 'Tikus',
        '6': 'Babi', '7': 'Anjing', '8': 'Ayam', '9': 'Monyet', '10': 'Kambing',
        '11': 'Kuda', '12': 'Ular'
    }
    day = days_mapping.get(units_digit, 'Tidak Diketahui')
    shio = shio_mapping.get(units_digit, 'Tidak Diketahui')
    return day, shio

def get_dates_for_days(days, input_date_str):
    """Dapatkan tanggal untuk hari tertentu dalam bulan dan tahun tertentu."""
    current_date = datetime.strptime(input_date_str, "%d-%m-%Y")
    current_month = current_date.month
    current_year = current_date.year

    dates_for_days = {day: [] for day in days}

    for day in range(1, 32):
        try:
            date = datetime(current_year, current_month, day)
            day_name = date.strftime('%A')
            day_name_id = {
                'Sunday': 'Minggu', 'Monday': 'Senin', 'Tuesday': 'Selasa',
                'Wednesday': 'Rabu', 'Thursday': 'Kamis', "Friday": "Jum'at", 'Saturday': 'Sabtu'
            }
            if day_name_id[day_name] in days:
                dates_for_days[day_name_id[day_name]].append(date.strftime('%Y-%m-%d'))
        except ValueError:
            continue

    return dates_for_days

def input_numbers():
    """Input nomor baru dari pengguna."""
    while True:
        number = input("Masukkan nomor 4D baru (atau ketik 'selesai' untuk menyelesaikan): ")
        if number.lower() == 'selesai':
            break
        elif len(number) == 4 and number.isdigit():
            if number not in numbers:
                numbers.append(number)
                save_numbers(numbers)
            else:
                print("Nomor sudah ada dalam daftar.")
        else:
            print("Nomor tidak valid. Harap masukkan nomor 4 digit.")

def process_digits(random_numbers):
    """Proses nomor acak untuk memisahkan, mengumpulkan, dan mengurutkan digit."""
    all_digits = []

    # Pisahkan digit dari setiap nomor acak dan tambahkan ke list
    for num in random_numbers:
        all_digits.extend(str(num))  # Tambahkan digit ke list
    
    # Hitung frekuensi kemunculan setiap digit
    digit_counter = Counter(all_digits)
    
    # Urutkan digit dan konversi kembali ke string
    sorted_digits = ''.join(sorted(digit_counter.keys()))
    
    return sorted_digits, digit_counter

def ai_think_mode(numbers, valid_last_digits):
    """Mode AI yang bisa berpikir sendiri, menyesuaikan prediksi berdasarkan pola dan randomness."""
    print("\n[AI Mode]: Mode berpikir sendiri diaktifkan.")
    
    # Menganalisis angka berdasarkan pola historis
    frequency = Counter(numbers)
    most_common = frequency.most_common(1)
    least_common = frequency.most_common()[:-2:-1]
    
    # Memadukan angka yang sering muncul dengan elemen acak
    ai_generated_numbers = []
    for _ in range(5):
        # Pilih angka dari yang paling sering muncul
        if most_common and random.random() > 0.7:
            number = random.choice(most_common)[0]
        else:
            # Jika tidak, pilih angka dari yang jarang muncul atau acak
            number = random.choice(least_common)[0] if random.random() > 0.3 else generate_random_number_for_days(numbers, valid_last_digits)
        
        ai_generated_numbers.append(number)
    
    return ai_generated_numbers

# Input tanggal dari pengguna
while True:
    input_date = input("Masukkan tanggal (DD-MM-YYYY): ")
    try:
        input_date_str = str(input_date)
        datetime.strptime(input_date_str, "%d-%m-%Y")
        break
    except ValueError:
        print("Format tanggal tidak valid. Harap masukkan tanggal dalam format DD-MM-YYYY.")

# Meminta pengguna untuk menambahkan tanggal khusus untuk hari-hari yang dipilih
days = ['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', "Jum'at", 'Sabtu']
dates_for_days = get_dates_for_days(days, input_date_str)

print("\nTanggal untuk hari-hari yang dipilih:")
for day, date_list in dates_for_days.items():
    print(f"{day}: {', '.join(date_list) if date_list else 'Tidak ada tanggal'}")

# Meminta pengguna untuk menambahkan nomor baru
input_numbers()

# Memproses data untuk model ML
X, y = prepare_data_for_ml(numbers)

# Melatih model
model, scaler = train_model(X, y)

# Prediksi nomor baru
valid_last_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
predicted_numbers = predict_with_model(model, scaler, numbers, valid_last_digits)

print(f"\nNomor yang diprediksi: {predicted_numbers}")

# Generate nomor acak berdasarkan hari tertentu
random_number = generate_random_number_for_days(numbers, valid_last_digits)
print(f"\nNomor acak berdasarkan hari tertentu: {random_number}")

# Memproses digit untuk analisis
sorted_digits, digit_counter = process_digits(predicted_numbers)
print(f"\nDigit yang terurut: {sorted_digits}")
print(f"Frekuensi kemunculan digit: {digit_counter}")

# Mengaktifkan mode AI
ai_generated_numbers = ai_think_mode(numbers, valid_last_digits)
print(f"\nNomor yang dihasilkan oleh AI: {ai_generated_numbers}")
