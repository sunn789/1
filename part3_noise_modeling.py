"""
بخش ۳: شناسایی و مدلسازی نویز
تمرینات درس پردازش گفتار - سری اول
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os
from scipy.fft import fft, fftfreq
from tabulate import tabulate

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def load_clean_signal():
    """
    بارگذاری سیگنال تمیز
    """
    try:
        sample_rate, audio_data = wav.read('sound1.wav')
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data / np.max(np.abs(audio_data))
        
    except FileNotFoundError:
        print("⚠️ فایل اصلی یافت نشد. سیگنال نمونه تولید می‌کنیم...")
        sample_rate = 44100
        duration = 3
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # تولید سیگنال نمونه
        freq1, freq2, freq3, freq4 = 100, 200, 300, 800
        audio_data = (0.5 * np.sin(2 * np.pi * freq1 * t) +
                      0.3 * np.sin(2 * np.pi * freq2 * t) +
                      0.2 * np.sin(2 * np.pi * freq3 * t) +
                      0.4 * np.sin(2 * np.pi * freq4 * t) *
                      np.exp(-0.5 * (t - duration/2)**2))
        
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    return sample_rate, audio_data

def generate_noise(noise_type, length, sample_rate):
    """
    تولید انواع مختلف نویز
    """
    if noise_type == 'gaussian':
        # نویز گوسی سفید
        noise = np.random.normal(0, 1, length)
        
    elif noise_type == 'babble':
        # نویز همهمه (شبیه محیط شلوغ)
        t = np.arange(length) / sample_rate
        noise = np.zeros(length)
        
        # ترکیب چندین سیگنال سینوسی با فرکانس‌های مختلف
        frequencies = np.linspace(100, 1000, 20)
        amplitudes = np.random.uniform(0.05, 0.2, 20)
        phases = np.random.uniform(0, 2*np.pi, 20)
        
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            noise += amp * np.sin(2 * np.pi * freq * t + phase)
            
        # اضافه کردن تغییرات زمانی
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        noise *= envelope
        
    elif noise_type == 'noisex':
        # نویز -92Noisex (شبیه نویز صنعتی)
        t = np.arange(length) / sample_rate
        
        # ترکیب نویز گوسی و ضربه‌ای
        gaussian = np.random.normal(0, 0.5, length)
        
        # ایجاد ضربه‌های تصادفی
        impulsive = np.zeros(length)
        num_impulses = int(length / 1000)
        impulse_positions = np.random.randint(0, length, num_impulses)
        impulse_amplitudes = np.random.uniform(0.5, 2.0, num_impulses)
        
        for pos, amp in zip(impulse_positions, impulse_amplitudes):
            impulsive[pos] = amp
            
        # فیلتر کردن ضربه‌ها
        from scipy.signal import lfilter
        b = [0.1, 0.2, 0.4, 0.2, 0.1]
        impulsive = lfilter(b, 1, impulsive)
        
        noise = gaussian + 0.3 * impulsive
        
    else:
        noise = np.zeros(length)
    
    # نرمال‌سازی
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    
    return noise

def add_noise_to_signal(signal, noise, target_snr_db):
    """
    اضافه کردن نویز به سیگنال با SNR مشخص
    """
    # محاسبه توان سیگنال و نویز
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    # تنظیم نویز برای دستیابی به SNR مورد نظر
    if noise_power > 0:
        # محاسبه توان نویز مورد نیاز
        target_noise_power = signal_power / (10**(target_snr_db/10))
        
        # تنظیم دامنه نویز
        scale_factor = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * scale_factor
    else:
        scaled_noise = np.zeros_like(noise)
    
    # ترکیب سیگنال و نویز
    noisy_signal = signal + scaled_noise
    
    # محاسبه SNR واقعی
    actual_noise = noisy_signal - signal
    actual_noise_power = np.mean(actual_noise**2)
    
    if actual_noise_power > 0:
        actual_snr_db = 10 * np.log10(signal_power / actual_noise_power)
    else:
        actual_snr_db = float('inf')
    
    return noisy_signal, scaled_noise, actual_snr_db

def calculate_snr(clean_signal, noisy_signal):
    """
    محاسبه SNR بین دو سیگنال
    """
    # محاسبه توان سیگنال تمیز
    signal_power = np.mean(clean_signal**2)
    
    # محاسبه توان نویز (تفاوت دو سیگنال)
    noise = noisy_signal - clean_signal
    noise_power = np.mean(noise**2)
    
    # محاسبه SNR (دسی‌بل)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return snr_db

def analyze_noise_characteristics(noise, sample_rate, noise_type):
    """
    تحلیل ویژگی‌های نویز
    """
    # محاسبه FFT نویز
    n = len(noise)
    fft_noise = fft(noise)
    fft_magnitude = np.abs(fft_noise[:n//2])
    frequencies = fftfreq(n, 1/sample_rate)[:n//2]
    
    # محاسبه آماری نویز
    stats = {
        'mean': np.mean(noise),
        'std': np.std(noise),
        'variance': np.var(noise),
        'max': np.max(noise),
        'min': np.min(noise),
        'rms': np.sqrt(np.mean(noise**2)),
        'crest_factor': np.max(np.abs(noise)) / np.sqrt(np.mean(noise**2)) if np.mean(noise**2) > 0 else 0
    }
    
    # محاسبه ویژگی‌های طیفی
    spectral_stats = {
        'peak_freq': frequencies[np.argmax(fft_magnitude)],
        'mean_freq': np.mean(frequencies),
        'bandwidth': frequencies[-1] - frequencies[0],
        'spectral_flatness': np.exp(np.mean(np.log(fft_magnitude + 1e-10))) / np.mean(fft_magnitude)
    }
    
    return stats, spectral_stats, frequencies, fft_magnitude

def plot_noise_comparison(clean_signal, noisy_signals, noise_types, sample_rate):
    """
    رسم مقایسه سیگنال‌های نویزی
    """
    fig, axes = plt.subplots(len(noise_types), 3, figsize=(15, 3*len(noise_types)))
    
    if len(noise_types) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (noise_type, noisy_signal) in enumerate(zip(noise_types, noisy_signals)):
        # زمان نمونه برای نمایش
        display_samples = min(2000, len(clean_signal))
        time_axis = np.arange(display_samples) / sample_rate
        
        # نمودار ۱: سیگنال تمیز
        axes[i, 0].plot(time_axis, clean_signal[:display_samples], color='blue', linewidth=1)
        axes[i, 0].set_title(f'سیگنال تمیز', fontsize=10)
        axes[i, 0].set_xlabel('زمان (ثانیه)', fontsize=8)
        axes[i, 0].set_ylabel('دامنه', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        
        # نمودار ۲: سیگنال نویزی
        axes[i, 1].plot(time_axis, noisy_signal[:display_samples], color='red', linewidth=1)
        axes[i, 1].set_title(f'سیگنال با نویز {noise_type}', fontsize=10)
        axes[i, 1].set_xlabel('زمان (ثانیه)', fontsize=8)
        axes[i, 1].set_ylabel('دامنه', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
        
        # نمودار ۳: نویز
        noise = noisy_signal[:display_samples] - clean_signal[:display_samples]
        axes[i, 2].plot(time_axis, noise, color='green', linewidth=1)
        axes[i, 2].set_title(f'نویز {noise_type} جدا شده', fontsize=10)
        axes[i, 2].set_xlabel('زمان (ثانیه)', fontsize=8)
        axes[i, 2].set_ylabel('دامنه', fontsize=8)
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output_images/part3_noisy_signals_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # نمودار طیف نویزها
    plt.figure(figsize=(14, 8))
    
    colors = ['red', 'green', 'blue']
    
    for i, noise_type in enumerate(noise_types):
        # تولید نویز برای تحلیل طیفی
        noise = generate_noise(noise_type, len(clean_signal), sample_rate)
        
        # محاسبه FFT
        n = len(noise)
        fft_noise = fft(noise)
        fft_magnitude = np.abs(fft_noise[:n//2])
        frequencies = fftfreq(n, 1/sample_rate)[:n//2]
        
        # رسم طیف
        plt.plot(frequencies[:n//4], 10*np.log10(fft_magnitude[:n//4] + 1e-10), 
                color=colors[i], linewidth=1, label=f'نویز {noise_type}', alpha=0.7)
    
    # رسم طیف سیگنال تمیز برای مقایسه
    n = len(clean_signal)
    fft_clean = fft(clean_signal)
    fft_magnitude_clean = np.abs(fft_clean[:n//2])
    frequencies = fftfreq(n, 1/sample_rate)[:n//2]
    
    plt.plot(frequencies[:n//4], 10*np.log10(fft_magnitude_clean[:n//4] + 1e-10), 
            color='black', linewidth=2, label='سیگنال تمیز', alpha=0.5)
    
    plt.title('مقایسه طیف فرکانسی انواع نویز با سیگنال تمیز', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('توان (dB)', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 4000)
    plt.savefig('output_images/part3_noise_spectrum_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_noise_analysis_results(noise_stats, snr_results):
    """
    ذخیره نتایج تحلیل نویز
    """
    # جدول ویژگی‌های آماری نویزها
    table1_data = []
    for noise_type, stats in noise_stats.items():
        table1_data.append([
            noise_type,
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats['variance']:.4f}",
            f"{stats['rms']:.4f}",
            f"{stats['crest_factor']:.2f}"
        ])
    
    # جدول نتایج SNR
    table2_data = []
    for noise_type, snr_info in snr_results.items():
        table2_data.append([
            noise_type,
            f"{snr_info['target_snr']:.1f} dB",
            f"{snr_info['actual_snr']:.2f} dB",
            f"{abs(snr_info['target_snr'] - snr_info['actual_snr']):.2f} dB"
        ])
    
    print("\n" + "="*70)
    print("نتایج تحلیل نویزها")
    print("="*70)
    
    print("\nویژگی‌های آماری نویزها:")
    print("-"*70)
    print(tabulate(table1_data, 
                   headers=["نوع نویز", "میانگین", "انحراف معیار", "واریانس", "RMS", "فاکتور قله"], 
                   tablefmt="grid"))
    
    print("\n\nنتایج SNR:")
    print("-"*70)
    print(tabulate(table2_data, 
                   headers=["نوع نویز", "SNR هدف", "SNR محاسبه شده", "خطا"], 
                   tablefmt="grid"))
    print("="*70)
    
    # ذخیره در فایل
    with open('part3_noise_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("نتایج شناسایی و مدلسازی نویز - بخش ۳\n")
        f.write("="*70 + "\n\n")
        
        f.write("ویژگی‌های آماری نویزها:\n")
        f.write("-"*70 + "\n")
        f.write(tabulate(table1_data, 
                         headers=["نوع نویز", "میانگین", "انحراف معیار", "واریانس", "RMS", "فاکتور قله"], 
                         tablefmt="simple"))
        f.write("\n\n")
        
        f.write("نتایج SNR:\n")
        f.write("-"*70 + "\n")
        f.write(tabulate(table2_data, 
                         headers=["نوع نویز", "SNR هدف", "SNR محاسبه شده", "خطا"], 
                         tablefmt="simple"))
        f.write("\n" + "="*70 + "\n")

def main():
    """
    تابع اصلی اجرای بخش ۳
    """
    print("="*70)
    print("بخش ۳: شناسایی و مدلسازی نویز")
    print("="*70)
    
    # ایجاد پوشه خروجی
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    
    # بارگذاری سیگنال تمیز
    sample_rate, clean_signal = load_clean_signal()
    
    # تنظیمات آزمایش
    target_snr_db = 10  # SNR هدف
    noise_types = ['gaussian', 'babble', 'noisex']
    
    # تولید نویزها و اضافه کردن به سیگنال
    noisy_signals = []
    noises = []
    snr_results = {}
    noise_stats = {}
    
    print(f"\n📊 تولید نویزها با SNR هدف: {target_snr_db} dB")
    print("-"*50)
    
    for noise_type in noise_types:
        print(f"\nدر حال پردازش نویز {noise_type}...")
        
        # تولید نویز
        noise = generate_noise(noise_type, len(clean_signal), sample_rate)
        
        # اضافه کردن نویز به سیگنال
        noisy_signal, scaled_noise, actual_snr = add_noise_to_signal(
            clean_signal, noise, target_snr_db
        )
        
        # تحلیل ویژگی‌های نویز
        stats, spectral_stats, _, _ = analyze_noise_characteristics(
            scaled_noise, sample_rate, noise_type
        )
        
        # ذخیره نتایج
        noisy_signals.append(noisy_signal)
        noises.append(scaled_noise)
        snr_results[noise_type] = {
            'target_snr': target_snr_db,
            'actual_snr': actual_snr
        }
        noise_stats[noise_type] = stats
        
        print(f"  ✓ SNR محاسبه شده: {actual_snr:.2f} dB")
        print(f"  ✓ انحراف معیار نویز: {stats['std']:.4f}")
        print(f"  ✓ فاکتور قله: {stats['crest_factor']:.2f}")
    
    # رسم نمودارهای مقایسه
    plot_noise_comparison(clean_signal, noisy_signals, noise_types, sample_rate)
    
    # ذخیره نتایج
    save_noise_analysis_results(noise_stats, snr_results)
    
    # توضیح انواع نویز
    print("\n" + "="*70)
    print("توضیح انواع نویز:")
    print("="*70)
    
    print("""
    ۱. نویز گوسی سفید:
       • ویژگی: توزیع نرمال، طیف فرکانسی یکنواخت
       • منبع: نویز الکترونیکی، نویز حرارتی
       • مدلسازی: با توزیع نرمال با میانگین صفر
       • تأثیر بر گفتار: تمام فرکانس‌ها را یکسان تحت تأثیر قرار می‌دهد
    
    ۲. نویز همهمه (Babble Noise):
       • ویژگی: ترکیب چندین منبع گفتار، شبیه محیط شلوغ
       • منبع: محیط‌های پرجمعیت، رستوران‌ها، سالن‌ها
       • مدلسازی: ترکیب سیگنال‌های سینوسی با فرکانس‌های مختلف
       • تأثیر بر گفتار: مشابه گفتار انسان، تشخیص را دشوار می‌کند
    
    ۳. نویز -92Noisex:
       • ویژگی: ترکیب نویز گوسی و ضربه‌ای، شبیه محیط صنعتی
       • منبع: ماشین‌آلات، تجهیزات الکتریکی، محیط‌های صنعتی
       • مدلسازی: ترکیب نویز گوسی با ضربه‌های تصادفی
       • تأثیر بر گفتار: باعث اعوجاج لحظه‌ای و از بین رفتن بخش‌هایی از سیگنال
    """)
    
    # اهمیت SNR
    print("\n" + "="*70)
    print("اهمیت SNR در پردازش گفتار:")
    print("="*70)
    
    print("""
    SNR (Signal-to-Noise Ratio) نسبت توان سیگنال به توان نویز است:
    
    فرمول: SNR(dB) = 10 × log10(توان سیگنال / توان نویز)
    
    مقادیر معمول SNR:
    • SNR > 30 dB: کیفیت عالی (استودیو)
    • SNR = 20-30 dB: کیفیت خوب (محیط آرام)
    • SNR = 10-20 dB: کیفیت متوسط (دفتر کار)
    • SNR < 10 dB: کیفیت ضعیف (خیابان شلوغ)
    
    در این آزمایش SNR هدف 10 dB انتخاب شده که شرایط محیط صنعتی پرسر و صدا را شبیه‌سازی می‌کند.
    
    محاسبه SNR:
    ۱. محاسبه توان سیگنال: P_signal = میانگین(سیگنال²)
    ۲. محاسبه توان نویز: P_noise = میانگین(نویز²)
    ۳. تبدیل به دسی‌بل: SNR_dB = 10 × log10(P_signal / P_noise)
    
    خطاهای محاسبه SNR:
    • خطای اندازه‌گیری: تفاوت بین SNR هدف و SNR محاسبه شده
    • دلیل: نویز تولید شده ممکن است دقیقاً توزیع آماری ایده‌ال نداشته باشد
    • اهمیت: در سیستم‌های واقعی، اندازه‌گیری دقیق SNR چالش‌برانگیز است
    """)
    
    print("\n✅ بخش ۳ با موفقیت تکمیل شد!")
    print("📊 تصاویر تولید شده:")
    print("   - output_images/part3_noisy_signals_comparison.png")
    print("   - output_images/part3_noise_spectrum_comparison.png")
    print("📄 فایل نتایج:")
    print("   - part3_noise_analysis_results.txt")
    print("💾 سیگنال‌های نویزی ذخیره شدند.")

if __name__ == "__main__":
    main()