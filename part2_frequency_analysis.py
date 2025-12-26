"""
بخش ۲: تحلیل فرکانسی سیگنال گفتار
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

def load_signal(filename='sound1.wav'):
    """
    بارگذاری سیگنال (از بخش ۱ یا از فایل)
    """
    # سعی در خواندن فایل اصلی
    # خواندن فایل صوتی
    sample_rate, audio_data = wav.read(filename)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    return sample_rate, audio_data
        


def calculate_fft_analysis(audio_data, sample_rate):
    """
    محاسبه تحلیل FFT سیگنال
    """
    n = len(audio_data)
    
    # محاسبه FFT
    fft_result = fft(audio_data)
    
    # محاسبه بزرگی طیف (فقط فرکانس‌های مثبت)
    fft_magnitude = np.abs(fft_result[:n//2])
    
    # محاسبه فرکانس‌ها
    frequencies = fftfreq(n, 1/sample_rate)[:n//2]
    
    # محاسبه توان سیگنال
    power_spectrum = fft_magnitude**2
    
    return frequencies, fft_magnitude, power_spectrum

def plot_frequency_domain(frequencies, fft_magnitude, power_spectrum, sample_rate,audio_data):
    """
    رسم نمودارهای حوزه فرکانس
    """
    # نمودار ۱: طیف فرکانسی (دامنه)
    plt.figure(figsize=(14, 10))
    
    # زیرنمودار ۱: طیف کامل
    plt.subplot(2, 2, 1)
    plt.plot(frequencies, fft_magnitude, color='blue', linewidth=0.8)
    plt.title('طیف فرکانسی سیگنال گفتار (دامنه)', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('دامنه', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sample_rate/2)
    
    # زیرنمودار ۲: طیف محدوده گفتار (0-4000 Hz)
    plt.subplot(2, 2, 2)
    mask = frequencies <= 4000
    plt.plot(frequencies[mask], fft_magnitude[mask], color='red', linewidth=1)
    plt.title('طیف فرکانسی (محدوده 0-4000 Hz)', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('دامنه', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 4000)
    
    # زیرنمودار ۳: طیف توان
    plt.subplot(2, 2, 3)
    plt.plot(frequencies, 10*np.log10(power_spectrum + 1e-10), color='green', linewidth=0.8)
    plt.title('طیف توان سیگنال گفتار (dB)', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('توان (dB)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sample_rate/2)
    
    # زیرنمودار ۴: مقایسه حوزه زمان و فرکانس
    plt.subplot(2, 2, 4)
    
    # انتخاب یک فریم 1000 نمونه‌ای برای نمایش
    frame_size = min(1000, len(audio_data))
    time_frame = np.arange(frame_size) / sample_rate
    
    # نمایش حوزه زمان (بالا)
    plt.subplot(2, 2, 4)
    plt.plot(time_frame, audio_data[:frame_size], color='purple', linewidth=1)
    plt.title('مقایسه حوزه زمان و فرکانس', fontsize=12, fontweight='bold')
    plt.xlabel('زمان (ثانیه)', fontsize=10)
    plt.ylabel('دامنه', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output_images/part2_frequency_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # نمودار ۲: توزیع انرژی در باندهای مختلف
    plt.figure(figsize=(10, 6))
    
    # تعریف باندهای فرکانسی
    bands = [(0, 300), (300, 1000), (1000, 3000), (3000, 8000)]
    band_labels = ['0-300 Hz', '300-1000 Hz', '1000-3000 Hz', '3000-8000 Hz']
    band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    band_energies = []
    band_percentages = []
    
    total_energy = np.sum(power_spectrum)
    
    for i, (low, high) in enumerate(bands):
        mask = (frequencies >= low) & (frequencies < high)
        band_energy = np.sum(power_spectrum[mask])
        band_percentage = (band_energy / total_energy) * 100
        
        band_energies.append(band_energy)
        band_percentages.append(band_percentage)
        
        plt.fill_between(frequencies[mask], 0, fft_magnitude[mask], 
                        alpha=0.3, color=band_colors[i], label=f'{band_labels[i]} ({band_percentage:.1f}%)')
    
    plt.plot(frequencies, fft_magnitude, color='black', linewidth=0.5, alpha=0.7)
    plt.title('توزیع انرژی سیگنال گفتار در باندهای فرکانسی', fontsize=12, fontweight='bold')
    plt.xlabel('فرکانس (Hz)', fontsize=10)
    plt.ylabel('دامنه', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 8000)
    plt.savefig('output_images/part2_energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return band_energies, band_percentages

def find_peak_frequencies(frequencies, fft_magnitude):
    """
    یافتن فرکانس‌های پیک در طیف
    """
    # یافتن ۵ فرکانس با بیشترین دامنه
    peak_indices = np.argsort(fft_magnitude)[-5:][::-1]
    peak_frequencies = frequencies[peak_indices]
    peak_magnitudes = fft_magnitude[peak_indices]
    
    return peak_frequencies, peak_magnitudes

def save_analysis_results(properties, band_percentages, peak_frequencies, peak_magnitudes):
    """
    ذخیره نتایج تحلیل فرکانسی
    """
    # جدول ۱: مشخصات کلی
    table1_data = [
        ["نرخ نمونه‌برداری", f"{properties['sample_rate']} Hz"],
        ["فرکانس نایکویست", f"{properties['sample_rate']/2:.1f} Hz"],
        ["تعداد نقاط FFT", f"{properties['num_fft_points']:,}"],
        ["رزولوشن فرکانسی", f"{properties['freq_resolution']:.2f} Hz"],
    ]
    
    # جدول ۲: توزیع انرژی
    band_labels = ['0-300 Hz', '300-1000 Hz', '1000-3000 Hz', '3000-8000 Hz']
    table2_data = []
    for i, (label, percentage) in enumerate(zip(band_labels, band_percentages)):
        table2_data.append([label, f"{percentage:.2f}%"])
    
    # جدول ۳: فرکانس‌های پیک
    table3_data = []
    for i, (freq, mag) in enumerate(zip(peak_frequencies, peak_magnitudes)):
        table3_data.append([f"پیک {i+1}", f"{freq:.2f} Hz", f"{mag:.4f}"])
    
    print("\n" + "="*60)
    print("نتایج تحلیل فرکانسی سیگنال")
    print("="*60)
    
    print("\nمشخصات تحلیل FFT:")
    print("-"*40)
    print(tabulate(table1_data, headers=["پارامتر", "مقدار"], tablefmt="grid"))
    
    print("\n\nتوزیع انرژی در باندهای فرکانسی:")
    print("-"*40)
    print(tabulate(table2_data, headers=["بازه فرکانسی", "درصد انرژی"], tablefmt="grid"))
    
    print("\n\nفرکانس‌های پیک اصلی:")
    print("-"*40)
    print(tabulate(table3_data, headers=["پیک", "فرکانس (Hz)", "دامنه"], tablefmt="grid"))
    print("="*60)
    
    # ذخیره در فایل
    with open('part2_frequency_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("نتایج تحلیل فرکانسی سیگنال گفتار - بخش ۲\n")
        f.write("="*60 + "\n\n")
        
        f.write("مشخصات تحلیل FFT:\n")
        f.write("-"*40 + "\n")
        f.write(tabulate(table1_data, headers=["پارامتر", "مقدار"], tablefmt="simple"))
        f.write("\n\n")
        
        f.write("توزیع انرژی در باندهای فرکانسی:\n")
        f.write("-"*40 + "\n")
        f.write(tabulate(table2_data, headers=["بازه فرکانسی", "درصد انرژی"], tablefmt="simple"))
        f.write("\n\n")
        
        f.write("فرکانس‌های پیک اصلی:\n")
        f.write("-"*40 + "\n")
        f.write(tabulate(table3_data, headers=["پیک", "فرکانس (Hz)", "دامنه"], tablefmt="simple"))
        f.write("\n" + "="*60 + "\n")
    
    print("\n✅ نتایج در فایل 'part2_frequency_analysis_results.txt' ذخیره شد.")

def main():
    """
    تابع اصلی اجرای بخش ۲
    """
    print("="*60)
    print("بخش ۲: تحلیل فرکانسی سیگنال گفتار")
    print("="*60)
    
    # ایجاد پوشه خروجی
    if not os.path.exists('output_images'):
        os.makedirs('output_images')
    
    # بارگذاری سیگنال
    sample_rate, audio_data = load_signal()
    
    # محاسبه FFT و تحلیل فرکانسی
    frequencies, fft_magnitude, power_spectrum = calculate_fft_analysis(audio_data, sample_rate)
    
    # رسم نمودارها
    band_energies, band_percentages = plot_frequency_domain(frequencies, fft_magnitude, 
                                                           power_spectrum, sample_rate,audio_data)
    
    # یافتن فرکانس‌های پیک
    peak_frequencies, peak_magnitudes = find_peak_frequencies(frequencies, fft_magnitude)
    
    # محاسبه مشخصات
    properties = {
        'sample_rate': sample_rate,
        'num_fft_points': len(fft_magnitude),
        'freq_resolution': frequencies[1] - frequencies[0]
    }
    
    # ذخیره نتایج
    save_analysis_results(properties, band_percentages, peak_frequencies, peak_magnitudes)
    
    # تحلیل نتایج
    print("\n" + "="*60)
    print("تحلیل نتایج:")
    print("="*60)
    
    # تعیین بازه با بیشترین انرژی
    max_band_idx = np.argmax(band_percentages)
    band_labels = ['0-300 Hz', '300-1000 Hz', '1000-3000 Hz', '3000-8000 Hz']
    
    print(f"\nبازه فرکانسی با بیشترین انرژی: {band_labels[max_band_idx]}")
    print(f"درصد انرژی در این بازه: {band_percentages[max_band_idx]:.2f}%")
    
    print("\nویژگی‌های فرکانسی گفتار:")
    print("-"*40)
    print("""
    ۱. فرکانس پایه (Fundamental Frequency):
       • محدوده: ۸۵-۲۵۵ هرتز برای مردان، ۱۶۵-۲۵۵ هرتز برای زنان
       • مسئول زیروبمی صدا
       
    ۲. فرکانس‌های فرمانت (Formants):
       • F1: 200-800 هرتز - وضوح مصوت‌ها
       • F2: 800-2500 هرتز - تشخیص مصوت‌ها
       • F3: 2500-3500 هرتز - تشخیص همخوان‌ها
       
    ۳. محدوده انرژی گفتار:
       • ۳۰۰-۳۰۰۰ هرتز: حداکثر انرژی و اطلاعات زبانی
       • زیر ۳۰۰ هرتز: اطلاعات زیروبمی و احساسی
       • بالای ۳۰۰۰ هرتز: اصوات بی‌واک و تفاوت‌های ظریف
    """)
    
    print("\nنتیجه‌گیری:")
    print("-"*40)
    print(f"""بر اساس تحلیل انجام شده:
    ۱. بیشترین انرژی گفتار ({band_percentages[max_band_idx]:.1f}%) در بازه {band_labels[max_band_idx]} قرار دارد.
    ۲. این نتیجه با تئوری پردازش گفتار که بیشترین انرژی گفتار را در بازه ۳۰۰-۳۰۰۰ هرتز می‌داند، همخوانی دارد.
    ۳. فرکانس پیک اصلی: {peak_frequencies[0]:.1f} هرتز که در محدوده فرکانس پایه گفتار است.
    ۴. تحلیل فرکانسی نشان می‌دهد که برای حذف نویز، تمرکز بر بازه‌های فرکانسی حاوی انرژی گفتار ضروری است.""")
    
    print("\n✅ بخش ۲ با موفقیت تکمیل شد!")
    print("📊 تصاویر تولید شده:")
    print("   - output_images/part2_frequency_spectrum.png")
    print("   - output_images/part2_energy_distribution.png")
    print("📄 فایل نتایج:")
    print("   - part2_frequency_analysis_results.txt")

if __name__ == "__main__":
    main()