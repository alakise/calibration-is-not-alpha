# EXECUTIVE SUMMARY

Bu faz Jev'e yeni ücretli çağrı yapmadan tamamlandı. Düzeltilmiş klasik breakout ailesi ve kısa canlı Kraken microstructure örneği aynı resmi maliyet modeliyle değerlendirildi. Sonuç: mevcut cutoff'ta maliyet sonrası doğrulanmış alpha adayı yok.

- Klasik veri: 86357 BTC 1m satır; {'S0_BREAKOUT': 18849, 'S1_BREAKOUT_VOL': 4898, 'S2_BREAKOUT_VOL_ER': 3386, 'S3_BREAKOUT_VOL_ER_VOLUME': 2213, 'S4_BREAKOUT_VOL_ER_VOLUME_ETH': 2124, 'MR0_MEAN_REVERSION': 6447}
- Microstructure cutoff: 2026-09-20T10:52:21+00:00; 26398119 raw satır, 4 reconnect, bad JSON=0
- Canonical state: {'PF_XBTUSD': 40724, 'PF_ETHUSD': 40347}; valid book rows={'PF_XBTUSD': 38190, 'PF_ETHUSD': 37679}
- Jev manifest yalnızca tasarım; budget_usd=0 ve no_inference_launched=true.

## 1. Corrected classical strategy results

15 dakika maksimum tutuş, 15 bps round-trip maliyet. Candidate count ile executed trade count ayrıdır.

| Strategy | Candidates | Trades | Gross USD | Net USD | Gross/trade bps | Net/trade bps |
|---|---:|---:|---:|---:|---:|---:|
| S0_BREAKOUT | 18,849 | 9,002 | -138.74 | -13,641.74 | -0.154 | -15.154 |
| S1_BREAKOUT_VOL | 4,898 | 2,733 | 4.82 | -4,094.68 | 0.018 | -14.982 |
| S2_BREAKOUT_VOL_ER | 3,386 | 1,877 | 12.80 | -2,802.70 | 0.068 | -14.932 |
| S3_BREAKOUT_VOL_ER_VOLUME | 2,213 | 1,433 | 12.26 | -2,137.24 | 0.086 | -14.914 |
| S4_BREAKOUT_VOL_ER_VOLUME_ETH | 2,124 | 1,371 | 22.47 | -2,034.03 | 0.164 | -14.836 |
| MR0_MEAN_REVERSION | 6,447 | 5,004 | 14.15 | -7,491.85 | 0.028 | -14.972 |

## 2. What changed after fixing S1 volatility scaling

Önceki S1 horizon-scaled volatilite karşılaştırması geçersizdi. Yeni tanım aynı frekansta `vol5/vol60`; S1 artık 4,898 aday ve 2,733 gerçekleşen trade üretiyor. Bu nedenle önceki tek-trade S1 sonucu kanıt olarak kullanılmıyor.

## 3–6. Breakout, ER, volume ve ETH

Katmanlar aynı timestamp/parametre evreninde nested. Brüt iyileşmeler görülebilse de resmi 15 bps maliyet altında tüm canonical katmanlar negatif. Walk-forward'da her validation fold flat başlıyor; seçilen S4/N60 fold netleri sırasıyla -$223.69, -$187.17, -$164.28.

## 7. Microstructure dataset audit

Raw feed: {'book': 26301831, 'book_snapshot': 12, 'trade': 96257, 'trade_snapshot': 12, 'info': 7}; symbols={'PF_ETHUSD': 9774195, 'PF_XBTUSD': 16623917, 'unknown': 7}; interval=2026-09-19T23:36:50+00:00 → 2026-09-20T10:52:21+00:00. Collector public-only kaldı. Trade side Kraken taker-side metadata olarak saklandı; yerel aggressor çıkarımı yapılmadı. L1/L10 yok, yalnızca L5 persisted; OFI gerçek price-level OFI değil depth-change proxy.

## 8–12. OFI, trade-flow, book, microprice ve ablation

Tam örneklem diagnostics içinde rank correlations: ofi_proxy_10s=0.0005, trade_flow_imbalance_10s=-0.0214, l5_imbalance=-0.0101, microprice_displacement_bps=-0.0101, spread_bps=0.0532. Geç validation target std=0 olduğundan M0–M5 OOS yön/korrelasyon metrikleri bilgi taşıyan hareket yokluğunda degenerate; bunlar predictive evidence değildir.

## 13. Directional strategy economics

Resmi maliyet 5 bps fee + 2 bps slippage + 0.5 bps spread her side; toplam 15 bps round-trip. Cost sensitivity 5/10/15/20 bps rapor JSON'da; gross edge, turnover ve maliyet burden birbirine karıştırılmadı.

## 14–16. Maker simulator, fill sensitivity, toxicity

Touch Q0: net 20.13 USD, fills 7,310. Conservative queue Q0: net 2.09 USD, fills 624. Q4'te fill sayısı dramatik biçimde düşüyor. Maker fee/rebate 0 varsayıldı; gerçek tier ve queue position bilinmediği için bunlar diagnostic, alpha iddiası değil.

## 17. Directional vs passive-maker

Directional branch resmi maliyette negatif. Passive-maker branch optimistic-touch'ta pozitif görünse de conservative queue, zero maker-fee ve tek kısa oturum varsayımlarına duyarlı; iki branch arasında kazanan ilan edilmiyor.

## 18–19. Proposed compact Jev meta-state / future bundle

Manifest; candidate type, normalized momentum, volatility regime, ER, volume confirmation, OFI windows, trade-flow, L1/L5 imbalance, microprice displacement, spread, depth ve liquidity state alanlarını öneriyor. Rejim, signal consistency, liquidity stress, toxic bid/ask ve quote environment typed judgments olarak tasarlanmış; yeni Jev inference çalıştırılmadı.

## 20. Recommended next experiment

Daha uzun ve fiyat hareketi varyasyonu olan microstructure cutoff'ı topla; aynı feature setini non-degenerate chronological blocks üzerinde yeniden test et; maker tarafında gerçek signed fee/rebate schedule ve daha conservative queue/cancel latency modeli ekle. Sonra yalnızca ön-kayıtlı, küçük bir Jev meta-gate holdout'u düşün.

## Explicit answers A–N

**A.** Evet, düzeltilmiş S1 brüt PnL S0'a göre iyileşti (4.82 vs -138.74 USD), ancak OOS net maliyet sonrası pozitif değil.

**B.** ER filtresi brüt sonucu 4.82 → 12.80 USD yaptı; bu tek başına istikrarlı maliyet-sonrası kanıt değil.

**C.** Volume filtresi bu koşuda brüt sonucu 12.80 → 12.26 USD yaptı; incremental değer OOS'ta doğrulanmadı.

**D.** ETH confirmation tam örneklemde brüt sonucu 12.26 → 22.47 USD artırdı; walk-forward'da maliyet sonrası hâlâ negatif.

**E.** OOS brüt toplam 8.35 USD; pozitif görünen katmanlar fold-stabil bir aday oluşturmuyor.

**F.** Hayır. Seçilen üç validation fold toplam neti -575.15 USD (15 bps round-trip).

**G.** Bu cutoff'ta OFI, trade-flow ve book imbalance kısa vadeli mid hareketinde güçlü/stabil OOS kanıt vermedi; son validation hedefi varyasyonsuzdu.

**H.** Tam örneklem tanısında spread en büyük rank korelasyonu gösterdi; bu likidite/örneklem etkisi olabilir, stabil incremental alpha değildir.

**I.** Gösterilmedi: microstructure nested modellerinin geç validation hedefi degenerate (std=0), bu nedenle directional iyileşme iddiası yapılamaz.

**J.** Optimistic touch pozitif görünüyor, fakat conservative queue Q0 2.09 USD ve maker ücreti 0 varsayımı kullanıyor; ekonomik kanıt değil.

**K.** Toxicity seviyeleri fill oranını düşürdü; kısa tek oturumda daha iyi net sonuç için güvenilir kanıt yok.

**L.** Şu an hiçbiri. Directional taraf 15 bps'de negatif; maker tarafı ise fee/queue varsayımlarına aşırı duyarlı.

**M.** Jev ancak deterministic modelin zaten hesaplamadığı rejim tutarlılığı, likidite stresi ve bid/ask toxicity meta-yargılarını ekleyebilir.

**N.** Hayır. Yeni ücretli Jev çağrısı için yeterli cost-surviving, OOS kanıt yok; yalnızca manifest hazırlandı.

## Limitations

- Collector L5 ile sınırlı; L1/L10, gerçek queue position, cancellation latency ve maker fee tier yok.
- Microstructure validation penceresinde hedef varyasyonu yok; bu nedenle model sonucu alpha kanıtı sayılmadı.
- Maker sonuçları bir kısa public-data oturumu ve zero-fee varsayımına dayanıyor.
- Gerçek Kraken emirleri ve yeni Jev çağrıları kapalı.
