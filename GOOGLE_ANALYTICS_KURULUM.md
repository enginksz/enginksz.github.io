# Google Analytics Kurulum Rehberi

Bu dosya, GitHub Pages sitenize Google Analytics ekleme adımlarını içerir.

## ✅ Yapılan İşlemler

1. ✅ Google Analytics desteği etkinleştirildi (`enable_google_analytics: true`)
2. ✅ Tracking kodu zaten mevcut (`_includes/scripts/analytics.liquid`)
3. ✅ Layout dosyasında analytics script'i dahil edilmiş

## 📋 Yapmanız Gerekenler

### Adım 1: Google Analytics Hesabı Oluşturma

1. [Google Analytics](https://analytics.google.com/) adresine gidin
2. Google hesabınızla giriş yapın
3. "Start measuring" (Ölçmeye Başla) butonuna tıklayın

### Adım 2: Hesap Oluşturma

1. Hesap adını girin (örn: "Kişisel Web Sitesi")
2. Hesap veri paylaşım ayarlarını seçin
3. "Next" (İleri) butonuna tıklayın

### Adım 3: Özellik (Property) Oluşturma

1. Özellik adını girin: **enginksz.github.io**
2. Raporlama zaman dilimini seçin (Türkiye için: UTC+3)
3. Para birimini seçin (TRY veya USD)
4. "Show advanced options" (Gelişmiş seçenekleri göster) tıklayın
5. "Create a Universal Analytics property" seçeneğini **KAPALI** bırakın (GA4 kullanacağız)
6. "Next" (İleri) butonuna tıklayın

### Adım 4: İşletme Bilgileri

1. İşletme sektörünü seçin (örn: "Technology" veya "Other")
2. İşletme büyüklüğünü seçin
3. Google Analytics'i nasıl kullanacağınızı seçin
4. "Create" (Oluştur) butonuna tıklayın
5. Kullanım şartlarını kabul edin

### Adım 5: Veri Akışı (Data Stream) Oluşturma

1. "Data Streams" (Veri Akışları) bölümüne gidin
2. "Add stream" (Akış Ekle) butonuna tıklayın
3. "Web" seçeneğini seçin
4. Web sitesi URL'nizi girin: **https://enginksz.github.io**
5. Stream adını girin (örn: "enginksz.github.io")
6. "Create stream" (Akış Oluştur) butonuna tıklayın

### Adım 6: Measurement ID'yi Kopyalama

1. Oluşturduğunuz web stream'in detay sayfasına gidin
2. **"Measurement ID"** değerini bulun (format: `G-XXXXXXXXXX`)
3. Bu ID'yi kopyalayın

### Adım 7: Measurement ID'yi Siteye Ekleme

1. `_config.yml` dosyasını açın
2. `google_analytics:` satırını bulun (yaklaşık 124. satır)
3. Measurement ID'nizi yapıştırın:

```yaml
google_analytics: G-XXXXXXXXXX  # Kendi ID'nizi buraya yapıştırın
```

### Adım 8: Değişiklikleri Yayınlama

1. Değişiklikleri commit edin:
   ```bash
   git add _config.yml
   git commit -m "Add Google Analytics tracking"
   git push
   ```

2. GitHub Pages otomatik olarak sitenizi yeniden oluşturacak (birkaç dakika sürebilir)

### Adım 9: Doğrulama

1. Sitenizi ziyaret edin: https://enginksz.github.io
2. Google Analytics'te "Realtime" (Gerçek Zamanlı) raporuna gidin
3. Birkaç saniye içinde kendi ziyaretinizi görmelisiniz

## 🔒 Gizlilik Notu

- Google Analytics tracking kodu **ziyaretçilere görünmez** - sadece arka planda çalışır
- Veriler sadece **sizin Google Analytics hesabınızda** görülebilir
- Ziyaretçiler tracking kodunun varlığını göremez (kaynak kodunu incelemedikleri sürece)

## 📊 Verileri Görüntüleme

Google Analytics'te şu bilgileri görebilirsiniz:
- Günlük/haftalık/aylık ziyaretçi sayıları
- Ziyaretçilerin geldiği ülkeler
- En çok ziyaret edilen sayfalar
- Ziyaretçilerin kullandığı cihazlar ve tarayıcılar
- Ziyaret süreleri
- Gerçek zamanlı ziyaretçi sayısı

## 🛠️ Sorun Giderme

### Tracking çalışmıyor mu?

1. `_config.yml` dosyasında `enable_google_analytics: true` olduğundan emin olun
2. `google_analytics:` alanına Measurement ID'nizi eklediğinizden emin olun
3. Değişikliklerin GitHub'a push edildiğinden emin olun
4. GitHub Pages'in sitenizi yeniden oluşturmasını bekleyin (2-5 dakika)
5. Tarayıcınızın geliştirici araçlarını açın (F12) ve Console'da hata olup olmadığını kontrol edin
6. Google Analytics'te "Realtime" raporunu kontrol edin (bazen 24 saat gecikme olabilir)

### Measurement ID formatı

- Doğru format: `G-XXXXXXXXXX` (G- ile başlar, ardından 10 karakter)
- Eski format (UA-): Artık desteklenmiyor, GA4 kullanın

## 📚 Ek Kaynaklar

- [Google Analytics Resmi Dokümantasyonu](https://support.google.com/analytics)
- [GA4 Başlangıç Rehberi](https://support.google.com/analytics/answer/9304153)
- [Jekyll ve Google Analytics](https://jekyllrb.com/docs/configuration/)

---

**Not:** Bu dosya sadece referans içindir. Kurulum tamamlandıktan sonra silmek isteyebilirsiniz.
