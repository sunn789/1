import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import os
from scipy.fft import fft, fftfreq
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')

# ایجاد پوشه برای ذخیره تصاویر
if not os.path.exists('output_images'):
    os.makedirs('output_images')

# 1a. خواندن فایل صوتی
try:
    # خواندن فایل صوتی
    sample_rate, audio_data = wav.read('sound1.wav')
    
    # اگر فایل استریو باشد، به مونو تبدیل می‌کنیم
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # نرمال‌سازی سیگنال
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    print(f"فایل صوتی با موفقیت خوانده شد.")
except FileNotFoundError:
    print("فایل sound1.wav یافت نشد. یک سیگنال نمونه تولید می‌کنیم...")
    # تولید یک سیگنال نمونه برای تست
    sample_rate = 16000  # نرخ نمونه‌برداری استاندارد برای گفتار
    duration = 3  # مدت زمان بر حسب ثانیه
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # تولید یک سیگنال گفتار شبیه‌سازی شده
    freq1 = 100
    freq2 = 200
    freq3 = 300
    freq4 = 800
    
    audio_data = (0.5 * np.sin(2 * np.pi * freq1 * t) +
                  0.3 * np.sin(2 * np.pi * freq2 * t) +
                  0.2 * np.sin(2 * np.pi * freq3 * t) +
                  0.4 * np.sin(2 * np.pi * freq4 * t) *
                  np.exp(-0.5 * (t - duration/2)**2))
    
    # نرمال‌سازی
    audio_data = audio_data / np.max(np.abs(audio_data))

# 1b. محاسبه طول سیگنال، نرخ نمونه‌برداری و تعداد نمونه‌ها
signal_length = len(audio_data) / sample_rate
num_samples = len(audio_data)

# ایجاد جدول برای مشخصات سیگنال
signal_info_table = [
    ["نرخ نمونه‌برداری", f"{sample_rate} Hz"],
    ["تعداد نمونه‌ها", f"{num_samples}"],
    ["طول سیگنال", f"{signal_length:.2f} ثانیه"]
]

print("\n" + "="*50)
print("جدول ۱: مشخصات سیگنال گفتار")
print("="*50)
print(tabulate(signal_info_table, headers=["پارامتر", "مقدار"], tablefmt="grid"))
print("="*50)

# ذخیره جدول در فایل
with open('signal_info_table.txt', 'w', encoding='utf-8') as f:
    f.write("جدول ۱: مشخصات سیگنال گفتار\n")
    f.write("="*50 + "\n")
    f.write(tabulate(signal_info_table, headers=["پارامتر", "مقدار"], tablefmt="simple"))
    f.write("\n" + "="*50 + "\n")

# 1a. نمایش سیگنال در حوزه زمان
plt.figure(figsize=(12, 6))
time_axis = np.arange(len(audio_data)) / sample_rate
plt.plot(time_axis, audio_data)
plt.title('سیگنال گفتار در حوزه زمان')
plt.xlabel('زمان (ثانیه)')
plt.ylabel('دامنه')
plt.grid(True)
plt.savefig('output_images/speech_signal_time_domain.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. تحلیل فرکانسی سیگنال گفتار
n = len(audio_data)
fft_result = fft(audio_data)
fft_magnitude = np.abs(fft_result[:n//2])
frequencies = fftfreq(n, 1/sample_rate)[:n//2]

# 2b. رسم نمودار طیف فرکانسی
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_magnitude)
plt.title('طیف فرکانسی سیگنال گفتار')
plt.xlabel('فرکانس (Hz)')
plt.ylabel('دامنه')
plt.grid(True)
plt.xlim(0, 4000)
plt.savefig('output_images/speech_signal_frequency_domain.png', dpi=300, bbox_inches='tight')
plt.show()

# تعیین بازه فرکانسی با بیشترین انرژی
max_energy_idx = np.argmax(fft_magnitude)
max_energy_freq = frequencies[max_energy_idx]

# محاسبه انرژی در بازه‌های فرکانسی مختلف
energy_low = np.sum(fft_magnitude[(frequencies >= 0) & (frequencies < 300)]**2)
energy_mid = np.sum(fft_magnitude[(frequencies >= 300) & (frequencies < 3000)]**2)
energy_high = np.sum(fft_magnitude[(frequencies >= 3000) & (frequencies < 8000)]**2)

total_energy = energy_low + energy_mid + energy_high

# ایجاد جدول برای توزیع انرژی فرکانسی
energy_distribution_table = [
    ["0-300 Hz", f"{energy_low/total_energy*100:.2f}%"],
    ["300-3000 Hz", f"{energy_mid/total_energy*100:.2f}%"],
    ["3000-8000 Hz", f"{energy_high/total_energy*100:.2f}%"],
    ["بیشترین انرژی", f"{max_energy_freq:.2f} Hz"]
]

print("\n" + "="*50)
print("جدول ۲: توزیع انرژی سیگنال در بازه‌های فرکانسی")
print("="*50)
print(tabulate(energy_distribution_table, headers=["بازه فرکانسی", "درصد انرژی"], tablefmt="grid"))
print("="*50)

# ذخیره جدول در فایل
with open('energy_distribution_table.txt', 'w', encoding='utf-8') as f:
    f.write("جدول ۲: توزیع انرژی سیگنال در بازه‌های فرکانسی\n")
    f.write("="*50 + "\n")
    f.write(tabulate(energy_distribution_table, headers=["بازه فرکانسی", "درصد انرژی"], tablefmt="simple"))
    f.write("\n" + "="*50 + "\n")

# 3. شناسایی و مدلسازی نویز
def add_noise(signal, noise_type, snr_db):
    signal_power = np.mean(signal**2)
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 1, len(signal))
    elif noise_type == 'babble':
        t = np.linspace(0, len(signal)/sample_rate, len(signal))
        noise = 0
        for freq in [100, 200, 300, 400, 500, 600, 700]:
            noise += 0.1 * np.sin(2 * np.pi * freq * t + np.random.rand()*2*np.pi)
    elif noise_type == 'noisex':
        noise = np.random.uniform(-1, 1, len(signal))
    else:
        noise = np.zeros(len(signal))
    
    noise_power = np.mean(noise**2)
    
    if noise_power > 0:
        target_noise_power = signal_power / (10**(snr_db/10))
        scale_factor = np.sqrt(target_noise_power / noise_power)
        noise = noise * scale_factor
    else:
        noise = np.zeros(len(signal))
    
    noisy_signal = signal + noise
    
    return noisy_signal, noise

def calculate_snr(original_signal, noisy_signal):
    signal_power = np.mean(original_signal**2)
    noise_signal = noisy_signal - original_signal
    noise_power = np.mean(noise_signal**2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return snr_db

# اضافه کردن نویزها با SNR 10dB
snr_target = 10

# نویز گوسی
noisy_gaussian, noise_gaussian = add_noise(audio_data, 'gaussian', snr_target)
snr_gaussian = calculate_snr(audio_data, noisy_gaussian)

# نویز همهمه
noisy_babble, noise_babble = add_noise(audio_data, 'babble', snr_target)
snr_babble = calculate_snr(audio_data, noisy_babble)

# نویز -92Noisex
noisy_noisex, noise_noisex = add_noise(audio_data, 'noisex', snr_target)
snr_noisex = calculate_snr(audio_data, noisy_noisex)

# ایجاد جدول برای SNR قبل و بعد از اضافه کردن نویز
snr_comparison_table = [
    ["بدون نویز", "نامحدود", "-"],
    ["نویز گوسی", f"{snr_gaussian:.2f} dB", f"{snr_target:.2f} dB"],
    ["نویز همهمه", f"{snr_babble:.2f} dB", f"{snr_target:.2f} dB"],
    ["نویز -92Noisex", f"{snr_noisex:.2f} dB", f"{snr_target:.2f} dB"]
]

print("\n" + "="*50)
print("جدول ۳: مقایسه SNR قبل و بعد از اضافه کردن نویز")
print("="*50)
print(tabulate(snr_comparison_table, 
               headers=["نوع سیگنال", "SNR محاسبه شده", "SNR هدف"], 
               tablefmt="grid"))
print("="*50)

# ذخیره جدول در فایل
with open('snr_comparison_table.txt', 'w', encoding='utf-8') as f:
    f.write("جدول ۳: مقایسه SNR قبل و بعد از اضافه کردن نویز\n")
    f.write("="*50 + "\n")
    f.write(tabulate(snr_comparison_table, 
                     headers=["نوع سیگنال", "SNR محاسبه شده", "SNR هدف"], 
                     tablefmt="simple"))
    f.write("\n" + "="*50 + "\n")

# نمایش سیگنال‌های نویزی
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# نویز گوسی
axes[0, 0].plot(time_axis[:1000], audio_data[:1000])
axes[0, 0].set_title('سیگنال اصلی (بخش اول)')
axes[0, 0].set_xlabel('زمان (ثانیه)')
axes[0, 0].set_ylabel('دامنه')
axes[0, 0].grid(True)

axes[0, 1].plot(time_axis[:1000], noisy_gaussian[:1000])
axes[0, 1].set_title(f'سیگنال با نویز گوسی (SNR={snr_gaussian:.2f}dB)')
axes[0, 1].set_xlabel('زمان (ثانیه)')
axes[0, 1].set_ylabel('دامنه')
axes[0, 1].grid(True)

# نویز همهمه
axes[1, 0].plot(time_axis[:1000], audio_data[:1000])
axes[1, 0].set_title('سیگنال اصلی (بخش اول)')
axes[1, 0].set_xlabel('زمان (ثانیه)')
axes[1, 0].set_ylabel('دامنه')
axes[1, 0].grid(True)

axes[1, 1].plot(time_axis[:1000], noisy_babble[:1000])
axes[1, 1].set_title(f'سیگنال با نویز همهمه (SNR={snr_babble:.2f}dB)')
axes[1, 1].set_xlabel('زمان (ثانیه)')
axes[1, 1].set_ylabel('دامنه')
axes[1, 1].grid(True)

# نویز -92Noisex
axes[2, 0].plot(time_axis[:1000], audio_data[:1000])
axes[2, 0].set_title('سیگنال اصلی (بخش اول)')
axes[2, 0].set_xlabel('زمان (ثانیه)')
axes[2, 0].set_ylabel('دامنه')
axes[2, 0].grid(True)

axes[2, 1].plot(time_axis[:1000], noisy_noisex[:1000])
axes[2, 1].set_title(f'سیگنال با نویز -92Noisex (SNR={snr_noisex:.2f}dB)')
axes[2, 1].set_xlabel('زمان (ثانیه)')
axes[2, 1].set_ylabel('دامنه')
axes[2, 1].grid(True)

plt.tight_layout()
plt.savefig('output_images/noisy_signals_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. حذف نویز و بازسازی سیگنال
def design_fir_filter(numtaps, cutoff_low, cutoff_high, fs):
    return signal.firwin(numtaps, [cutoff_low, cutoff_high], 
                         pass_zero=False, fs=fs, window='hann')

def design_iir_filter(order, cutoff_low, cutoff_high, fs, filter_type='butter'):
    nyquist = fs / 2
    low = cutoff_low / nyquist
    high = cutoff_high / nyquist
    
    if filter_type == 'butter':
        b, a = signal.butter(order, [low, high], btype='band')
    elif filter_type == 'cheby1':
        b, a = signal.cheby1(order, 0.5, [low, high], btype='band')
    elif filter_type == 'cheby2':
        b, a = signal.cheby2(order, 40, [low, high], btype='band')
    elif filter_type == 'ellip':
        b, a = signal.ellip(order, 0.5, 40, [low, high], btype='band')
    else:
        b, a = signal.butter(order, [low, high], btype='band')
    
    return b, a

# پارامترهای فیلتر
numtaps_fir = 101
order_iir = 4
cutoff_low = 80
cutoff_high = 3800

# طراحی فیلتر FIR
fir_coeff = design_fir_filter(numtaps_fir, cutoff_low, cutoff_high, sample_rate)

# طراحی فیلترهای IIR مختلف
iir_butter_b, iir_butter_a = design_iir_filter(order_iir, cutoff_low, cutoff_high, sample_rate, 'butter')
iir_cheby1_b, iir_cheby1_a = design_iir_filter(order_iir, cutoff_low, cutoff_high, sample_rate, 'cheby1')
iir_cheby2_b, iir_cheby2_a = design_iir_filter(order_iir, cutoff_low, cutoff_high, sample_rate, 'cheby2')
iir_ellip_b, iir_ellip_a = design_iir_filter(order_iir, cutoff_low, cutoff_high, sample_rate, 'ellip')

# نمایش پاسخ فرکانسی فیلترها
w_fir, h_fir = signal.freqz(fir_coeff, fs=sample_rate)
w_butter, h_butter = signal.freqz(iir_butter_b, iir_butter_a, fs=sample_rate)
w_cheby1, h_cheby1 = signal.freqz(iir_cheby1_b, iir_cheby1_a, fs=sample_rate)

plt.figure(figsize=(12, 8))
plt.plot(w_fir, 20*np.log10(np.abs(h_fir)), label='FIR (هان)')
plt.plot(w_butter, 20*np.log10(np.abs(h_butter)), label='IIR باترورث')
plt.plot(w_cheby1, 20*np.log10(np.abs(h_cheby1)), label='IIR چبیشف نوع 1')
plt.title('پاسخ فرکانسی فیلترها')
plt.xlabel('فرکانس (Hz)')
plt.ylabel('دامنه (dB)')
plt.legend()
plt.grid(True)
plt.xlim(0, 5000)
plt.ylim(-60, 5)
plt.savefig('output_images/filters_frequency_response.png', dpi=300, bbox_inches='tight')
plt.show()

def apply_filter(signal_data, filter_type, filter_params):
    if filter_type == 'fir':
        coeff = filter_params
        filtered = signal.lfilter(coeff, 1.0, signal_data)
    elif filter_type == 'iir_butter':
        b, a = filter_params
        filtered = signal.lfilter(b, a, signal_data)
    elif filter_type == 'iir_cheby1':
        b, a = filter_params
        filtered = signal.lfilter(b, a, signal_data)
    elif filter_type == 'iir_cheby2':
        b, a = filter_params
        filtered = signal.lfilter(b, a, signal_data)
    elif filter_type == 'iir_ellip':
        b, a = filter_params
        filtered = signal.lfilter(b, a, signal_data)
    else:
        filtered = signal_data
    
    return filtered

filter_types = ['fir', 'iir_butter', 'iir_cheby1', 'iir_cheby2', 'iir_ellip']
filter_params = {
    'fir': fir_coeff,
    'iir_butter': (iir_butter_b, iir_butter_a),
    'iir_cheby1': (iir_cheby1_b, iir_cheby1_a),
    'iir_cheby2': (iir_cheby2_b, iir_cheby2_a),
    'iir_ellip': (iir_ellip_b, iir_ellip_a)
}

noisy_signals = {
    'gaussian': noisy_gaussian,
    'babble': noisy_babble,
    'noisex': noisy_noisex
}

# محاسبه نتایج
results = {}

for noise_type, noisy_signal in noisy_signals.items():
    results[noise_type] = {}
    original_snr = calculate_snr(audio_data, noisy_signal)
    results[noise_type]['original'] = original_snr
    
    for filter_type in filter_types:
        filtered_signal = apply_filter(noisy_signal, filter_type, filter_params[filter_type])
        improved_snr = calculate_snr(audio_data, filtered_signal)
        results[noise_type][filter_type] = improved_snr

# ایجاد جدول برای مقایسه عملکرد فیلترها
filter_performance_table = []
for noise_type in noisy_signals.keys():
    row = [noise_type, f"{results[noise_type]['original']:.2f}"]
    for filter_type in filter_types:
        row.append(f"{results[noise_type][filter_type]:.2f}")
    filter_performance_table.append(row)

headers = ["نوع نویز", "SNR اولیه"] + [
    "FIR", "IIR باترورث", "IIR چبیشف۱", "IIR چبیشف۲", "IIR بیضوی"
]

print("\n" + "="*70)
print("جدول ۴: مقایسه عملکرد فیلترها در بهبود SNR (مقادیر بر حسب dB)")
print("="*70)
print(tabulate(filter_performance_table, headers=headers, tablefmt="grid", floatfmt=".2f"))
print("="*70)

# ذخیره جدول در فایل
with open('filter_performance_table.txt', 'w', encoding='utf-8') as f:
    f.write("جدول ۴: مقایسه عملکرد فیلترها در بهبود SNR (مقادیر بر حسب dB)\n")
    f.write("="*70 + "\n")
    f.write(tabulate(filter_performance_table, headers=headers, tablefmt="simple", floatfmt=".2f"))
    f.write("\n" + "="*70 + "\n")

# نمایش گرافیکی نتایج
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, noise_type in enumerate(noisy_signals.keys()):
    # داده‌ها برای نمودار
    filter_names = ['اصلی'] + filter_types
    snr_values = [results[noise_type]['original']] + [results[noise_type][ft] for ft in filter_types]
    
    axes[idx].bar(filter_names, snr_values)
    axes[idx].set_title(f'عملکرد فیلترها برای نویز {noise_type}')
    axes[idx].set_xlabel('نوع فیلتر')
    axes[idx].set_ylabel('SNR (dB)')
    axes[idx].grid(True, axis='y')
    axes[idx].tick_params(axis='x', rotation=45)
    
    # اضافه کردن مقدار عددی روی هر میله
    for i, v in enumerate(snr_values):
        axes[idx].text(i, v + 0.1, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig('output_images/filters_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ایجاد جدول برای بهبود SNR
improvement_table = []
for noise_type in noisy_signals.keys():
    row = [noise_type]
    best_filter = None
    best_improvement = -float('inf')
    
    for filter_type in filter_types:
        improvement = results[noise_type][filter_type] - results[noise_type]['original']
        row.append(f"{improvement:.2f}")
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_filter = filter_type
    
    row.append(best_filter)
    row.append(f"{best_improvement:.2f}")
    improvement_table.append(row)

improvement_headers = ["نوع نویز", "FIR", "IIR باترورث", "IIR چبیشف۱", 
                       "IIR چبیشف۲", "IIR بیضوی", "بهترین فیلتر", "بهترین بهبود"]

print("\n" + "="*70)
print("جدول ۵: بهبود SNR برای هر فیلتر (مقادیر بر حسب dB)")
print("="*70)
print(tabulate(improvement_table, headers=improvement_headers, tablefmt="grid"))
print("="*70)

# ذخیره جدول در فایل
with open('improvement_table.txt', 'w', encoding='utf-8') as f:
    f.write("جدول ۵: بهبود SNR برای هر فیلتر (مقادیر بر حسب dB)\n")
    f.write("="*70 + "\n")
    f.write(tabulate(improvement_table, headers=improvement_headers, tablefmt="simple"))
    f.write("\n" + "="*70 + "\n")

# خلاصه نتایج
summary_table = []
for noise_type in noisy_signals.keys():
    best_filter = None
    best_snr = -float('inf')
    
    for filter_type in filter_types:
        if results[noise_type][filter_type] > best_snr:
            best_snr = results[noise_type][filter_type]
            best_filter = filter_type
    
    improvement = best_snr - results[noise_type]['original']
    summary_table.append([
        noise_type,
        f"{results[noise_type]['original']:.2f}",
        best_filter,
        f"{best_snr:.2f}",
        f"{improvement:.2f}"
    ])

print("\n" + "="*60)
print("جدول ۶: خلاصه نتایج - بهترین فیلتر برای هر نوع نویز")
print("="*60)
print(tabulate(summary_table, 
               headers=["نوع نویز", "SNR اولیه", "بهترین فیلتر", "SNR نهایی", "بهبود"], 
               tablefmt="grid"))
print("="*60)

# ذخیره جدول خلاصه
with open('summary_table.txt', 'w', encoding='utf-8') as f:
    f.write("جدول ۶: خلاصه نتایج - بهترین فیلتر برای هر نوع نویز\n")
    f.write("="*60 + "\n")
    f.write(tabulate(summary_table, 
                     headers=["نوع نویز", "SNR اولیه", "بهترین فیلتر", "SNR نهایی", "بهبود"], 
                     tablefmt="simple"))
    f.write("\n" + "="*60 + "\n")

# نمایش نمونه‌ای از سیگنال‌های فیلتر شده
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for idx, noise_type in enumerate(noisy_signals.keys()):
    # سطر اول: سیگنال اصلی
    axes[idx, 0].plot(time_axis[:1000], audio_data[:1000])
    axes[idx, 0].set_title(f'سیگنال اصلی')
    axes[idx, 0].set_xlabel('زمان')
    axes[idx, 0].set_ylabel('دامنه')
    axes[idx, 0].grid(True)
    
    # سطر دوم: سیگنال نویزی
    axes[idx, 1].plot(time_axis[:1000], noisy_signals[noise_type][:1000])
    axes[idx, 1].set_title(f'سیگنال با نویز {noise_type}')
    axes[idx, 1].set_xlabel('زمان')
    axes[idx, 1].set_ylabel('دامنه')
    axes[idx, 1].grid(True)
    
    # سطر سوم: سیگنال فیلتر شده با FIR
    filtered = apply_filter(noisy_signals[noise_type], 'fir', filter_params['fir'])
    axes[idx, 2].plot(time_axis[:1000], filtered[:1000])
    axes[idx, 2].set_title(f'فیلتر شده با FIR (SNR={results[noise_type]["fir"]:.2f}dB)')
    axes[idx, 2].set_xlabel('زمان')
    axes[idx, 2].set_ylabel('دامنه')
    axes[idx, 2].grid(True)

plt.tight_layout()
plt.savefig('output_images/filtered_signals_samples.png', dpi=300, bbox_inches='tight')
plt.show()

# ذخیره سیگنال‌های فیلتر شده
print("\nذخیره سیگنال‌های فیلتر شده...")
for noise_type in noisy_signals.keys():
    for filter_type in filter_types:
        filtered_signal = apply_filter(noisy_signals[noise_type], filter_type, filter_params[filter_type])
        filename = f'filtered_{noise_type}_{filter_type}.wav'
        # نرمال‌سازی قبل از ذخیره
        filtered_normalized = filtered_signal / np.max(np.abs(filtered_signal))
        wav.write(filename, sample_rate, (filtered_normalized * 32767).astype(np.int16))
        print(f"  {filename} ذخیره شد.")

print("\n" + "="*70)
print("خلاصه تصاویر تولید شده:")
print("="*70)
print("1. output_images/speech_signal_time_domain.png - سیگنال در حوزه زمان")
print("2. output_images/speech_signal_frequency_domain.png - طیف فرکانسی")
print("3. output_images/noisy_signals_comparison.png - مقایسه سیگنال‌های نویزی")
print("4. output_images/filters_frequency_response.png - پاسخ فرکانسی فیلترها")
print("5. output_images/filters_performance_comparison.png - عملکرد فیلترها")
print("6. output_images/filtered_signals_samples.png - نمونه‌های سیگنال‌های فیلتر شده")

print("\n" + "="*70)
print("خلاصه جداول تولید شده:")
print("="*70)
print("1. signal_info_table.txt - مشخصات سیگنال")
print("2. energy_distribution_table.txt - توزیع انرژی فرکانسی")
print("3. snr_comparison_table.txt - مقایسه SNR")
print("4. filter_performance_table.txt - عملکرد فیلترها")
print("5. improvement_table.txt - بهبود SNR")
print("6. summary_table.txt - خلاصه نتایج")

print("\n" + "="*70)
print("خلاصه فایل‌های صوتی تولید شده:")
print("="*70)
print("15 فایل صوتی فیلتر شده با نام‌های filtered_[نوع_نویز]_[نوع_فیلتر].wav")

print("\nپروژه با موفقیت تکمیل شد!")
print("تمام خروجی‌ها در پوشه پروژه ذخیره شدند.")