"""A fictional holding company's policy knowledge base: 36 sections in 6 areas.

Every section carries two natural-language questions that are *paraphrases* (different words
than the policy text, as real employees ask), each with the exact fact that answers it. That
gives ground truth for retrieval, answer correctness and citation checks. Twelve more
questions concern topics that are NOT in the knowledge base, to test abstention.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: str
    area: str
    title: str
    text: str
    qa: tuple[tuple[str, str], tuple[str, str]]  # (question, answer value that must appear in the text)


FACTS: list[Fact] = [
    # --- Bilgi Teknolojileri
    Fact("bt-laptop", "BT", "Dizüstü Bilgisayar Yenileme",
         "Çalışanların taşınabilir bilgisayarları dört yılda bir yenilenir. Yenileme talebi BT hizmet masası üzerinden açılır ve cihaz teslimi ortalama 5 iş günü sürer.",
         (("Laptopum ne zaman yenilenir?", "dört yıl"), ("Yeni cihaz talebi açtıktan sonra kaç günde elime geçer?", "5 iş günü"))),
    Fact("bt-vpn", "BT", "Uzak Erişim (VPN)",
         "Şirket ağına dışarıdan bağlanmak için kurumsal VPN istemcisi ve donanım güvenlik anahtarı gerekir. Oturumlar 12 saat sonra otomatik sonlanır.",
         (("Evden şirket sistemlerine bağlanırken oturumum ne kadar açık kalır?", "12 saat"), ("Dışarıdan bağlanmak için hangi ek cihaza ihtiyacım var?", "donanım güvenlik anahtarı"))),
    Fact("bt-yazilim", "BT", "Yazılım Lisans Talebi",
         "Onaylı katalog dışındaki yazılımlar için lisans talebi BT güvenlik ekibinin incelemesine gönderilir. İnceleme süresi en fazla 10 iş günüdür ve yıllık 2.000 TL üzerindeki lisanslar departman direktörü onayı ister.",
         (("Katalogda olmayan bir program kurdurmak istersem güvenlik incelemesi ne kadar sürer?", "10 iş günü"), ("Hangi tutarın üstündeki programlar için üst düzey yönetici imzası gerekir?", "2.000 TL"))),
    Fact("bt-yedek", "BT", "Veri Yedekleme",
         "Dosya sunucularının yedeği her gece 02:00'de alınır ve 90 gün saklanır. Silinen bir dosyanın geri getirilmesi için BT hizmet masasına talep açılması yeterlidir.",
         (("Yanlışlıkla sildiğim dosyayı kaç gün öncesine kadar kurtarabilirim?", "90 gün"), ("Sunucu kopyaları hangi saatte çıkarılıyor?", "02:00"))),
    Fact("bt-kesinti", "BT", "Servis Kesintisi Bildirimi",
         "Kritik sistemlerde planlı bakım en az 7 gün önceden duyurulur. Planlanmayan kesintilerde ilk bilgilendirme 30 dakika içinde yapılır.",
         (("Bakım çalışmaları için önceden kaç gün haber verilir?", "7 gün"), ("Beklenmedik bir arıza olursa ne kadar sürede bilgi verilir?", "30 dakika"))),
    Fact("bt-mobil", "BT", "Mobil Cihaz Yönetimi",
         "Kişisel telefonlardan kurumsal e-postaya erişim için mobil cihaz yönetimi uygulaması kurulmalıdır. Cihaz kaybında uzaktan silme işlemi 1 saat içinde başlatılır.",
         (("Telefonum çalınırsa verilerin uzaktan silinmesi ne kadar sürede başlar?", "1 saat"), ("Özel telefonumda iş postası kullanabilmem için ne kurmam lazım?", "mobil cihaz yönetimi uygulaması"))),
    # --- Satın Alma
    Fact("sa-limit", "Satın Alma", "Satın Alma Onay Limitleri",
         "50.000 TL'ye kadar olan alımlar departman müdürü, 50.000 ile 500.000 TL arasındaki alımlar direktör, 500.000 TL üzerindeki alımlar genel müdür yardımcısı onayı gerektirir.",
         (("300.000 TL'lik bir alım için kimin imzası lazım?", "direktör"), ("Hangi tutardan sonra üst yönetimden onay almak gerekir?", "500.000 TL"))),
    Fact("sa-teklif", "Satın Alma", "Teklif Toplama",
         "10.000 TL üzerindeki alımlarda en az üç yazılı fiyat teklifi toplanır. Tek tedarikçili alımlar gerekçe formuyla yönetilir.",
         (("Kaç farklı firmadan fiyat almam gerekiyor?", "üç"), ("Sadece bir firmadan alım yapılacaksa ne doldurulur?", "gerekçe formu"))),
    Fact("sa-odeme", "Satın Alma", "Tedarikçi Ödeme Vadesi",
         "Tedarikçi faturaları standart olarak 45 gün vadeyle ödenir. Küçük işletme statüsündeki tedarikçilere vade 30 güne indirilir.",
         (("Fatura bedelleri kaç gün sonra hesaba geçer?", "45 gün"), ("Küçük firmalar için ödeme süresi daha kısa mı?", "30 gün"))),
    Fact("sa-cerceve", "Satın Alma", "Çerçeve Sözleşmeler",
         "Yıllık harcaması 1 milyon TL'yi aşan tedarikçilerle çerçeve sözleşme imzalanır. Sözleşme yenilemeleri bitiş tarihinden 90 gün önce başlatılır.",
         (("Yıllık hacmi hangi sınırı geçen firmalarla uzun dönemli anlaşma yapılır?", "1 milyon TL"), ("Anlaşma bitmeden kaç gün önce yenileme sürecine girilir?", "90 gün"))),
    Fact("sa-puan", "Satın Alma", "Tedarikçi Değerlendirme",
         "Kritik tedarikçiler yılda bir kez kalite, teslimat ve fiyat kriterlerine göre puanlanır. 60 puanın altındaki tedarikçiler iyileştirme planına alınır.",
         (("Hangi puanın altında kalan satıcılardan iyileştirme istenir?", "60 puan"), ("Önemli firmaların performansı hangi sıklıkla ölçülür?", "yılda bir"))),
    Fact("sa-acil", "Satın Alma", "Acil Satın Alma",
         "Acil alım talepleri satın alma müdürünün onayıyla 24 saat içinde işleme alınır. Acil statüsü yılda en fazla 3 kez aynı departman için kullanılabilir.",
         (("Acil siparişler ne kadar sürede başlatılır?", "24 saat"), ("Bir birim yıl içinde kaç kez acil statüsü isteyebilir?", "3 kez"))),
    # --- Finans
    Fact("fi-masraf", "Finans", "Masraf Raporu",
         "Seyahat sonrası masraf raporu 15 gün içinde sisteme girilmelidir. Geç girilen raporlar bir sonraki bordro dönemine kalır.",
         (("Yolculuktan döndükten sonra harcama bildirimini en geç kaç günde yapmalıyım?", "15 gün"), ("Geç yaptığım bildirimin parası ne zaman gelir?", "bir sonraki bordro dönemi"))),
    Fact("fi-avans", "Finans", "Seyahat Avansı",
         "Avans talepleri seyahatten en az 5 iş günü önce yapılır ve en fazla tahmini masrafın %80'ini kapsar. Avans, dönüşten sonraki 10 gün içinde kapatılmalıdır.",
         (("Yolculuk öncesi ön ödemenin üst sınırı nedir?", "%80"), ("Verilen ön ödeme dönüşten sonra kaç günde kapatılmalı?", "10 gün"))),
    Fact("fi-butce", "Finans", "Bütçe Revizyonu",
         "Departman bütçeleri yılda iki kez, Nisan ve Ekim aylarında revize edilir. %10'u aşan sapmalar finans komitesine gerekçeyle sunulur.",
         (("Bütçe güncellemeleri hangi aylarda yapılır?", "Nisan ve Ekim"), ("Planlanandan ne kadar fazla harcama yapılırsa komiteye açıklama gerekir?", "%10"))),
    Fact("fi-kur", "Finans", "Döviz İşlemleri",
         "Döviz cinsinden faturalar ödeme günündeki Merkez Bankası efektif satış kuruyla kayda alınır. 100.000 Avro üzeri işlemler için hazine birimi önceden bilgilendirilir.",
         (("Yabancı para faturaları hangi kurla muhasebeleştirilir?", "efektif satış"), ("Hangi tutarın üzerinde döviz işlemi hazineye önceden haber verilir?", "100.000 Avro"))),
    Fact("fi-kapanis", "Finans", "Ay Sonu Kapanış",
         "Aylık muhasebe kapanışı ayın 5. iş günü tamamlanır. Kapanıştan sonra geçmiş döneme fiş kesilmesi için finans müdürü onayı gerekir.",
         (("Ay sonu kayıtları kaçıncı iş gününde kilitlenir?", "5. iş günü"), ("Kapanmış döneme sonradan kayıt eklemek için kimin izni gerekir?", "finans müdürü"))),
    Fact("fi-kart", "Finans", "Kurumsal Kart Kullanımı",
         "Kurumsal kartlar yalnızca iş amaçlı harcamalar içindir ve aylık limit 20.000 TL'dir. Ekstre kalemleri ayın 25'ine kadar belgelendirilmelidir.",
         (("Şirket kartının aylık harcama sınırı ne kadar?", "20.000 TL"), ("Kart ekstresindeki harcamaların belgeleri ne zamana kadar yüklenmeli?", "ayın 25'ine"))),
    # --- Hukuk ve Uyum
    Fact("hu-saklama", "Hukuk", "Belge Saklama Süreleri",
         "Ticari defterler ve faturalar 10 yıl, çalışan özlük dosyaları iş sözleşmesinin bitiminden sonra 10 yıl saklanır. Saklama süresi dolan belgeler kontrollü imha edilir.",
         (("Faturaların arşivde tutulma süresi nedir?", "10 yıl"), ("Süresi dolan evraklara ne yapılır?", "kontrollü imha"))),
    Fact("hu-kvkk", "Hukuk", "Kişisel Veri Talepleri",
         "İlgili kişilerin başvuruları en geç 30 gün içinde yanıtlanır. Başvurular uyum birimine kvkk@atlas.example adresinden iletilir.",
         (("Kişisel veri başvurularına en geç kaç günde dönülür?", "30 gün"), ("Veri sahibi talepleri hangi e-posta adresine yönlendirilir?", "kvkk@atlas.example"))),
    Fact("hu-hediye", "Hukuk", "Hediye ve Ağırlama",
         "İş ilişkisi içinde 1.500 TL'yi aşan hediyeler kabul edilemez ve uyum birimine bildirilir. Tedarikçi yemek davetleri yıl içinde en fazla iki kez kabul edilebilir.",
         (("Bir iş ortağından gelen armağan hangi değerden sonra reddedilmeli?", "1.500 TL"), ("Satıcı firmaların yemek davetini yılda en çok kaç defa kabul edebilirim?", "iki kez"))),
    Fact("hu-etik", "Hukuk", "Etik İhbar Hattı",
         "Etik ihlaller 7/24 çalışan ihbar hattına isim vermeden bildirilebilir. İhbarlar 5 iş günü içinde değerlendirmeye alınır ve ihbar edene misilleme yapılamaz.",
         (("Kimliğimi gizleyerek usulsüzlük bildirebilir miyim?", "isim vermeden"), ("Bildirimim kaç günde incelemeye alınır?", "5 iş günü"))),
    Fact("hu-sozlesme", "Hukuk", "Sözleşme İnceleme",
         "Hukuk müşavirliği standart dışı sözleşmeleri 7 iş günü içinde inceler. 250.000 TL üzerindeki sözleşmelerde hukuk onayı zorunludur.",
         (("Bir anlaşma taslağını hukuka gönderirsem kaç günde dönüş alırım?", "7 iş günü"), ("Hangi bedelin üstünde sözleşmeler için hukuk imzası şart?", "250.000 TL"))),
    Fact("hu-cikar", "Hukuk", "Çıkar Çatışması",
         "Çalışanlar yakınlarının sahibi olduğu şirketlerle yapılacak işlemleri yöneticilerine yazılı olarak bildirir. Bildirim yapılmaması disiplin sürecini başlatır.",
         (("Akrabamın firmasıyla çalışılacaksa kime ve nasıl haber vermeliyim?", "yöneticilerine yazılı"), ("Aile şirketi bildirilmezse ne olur?", "disiplin süreci"))),
    # --- Güvenlik ve Tesis
    Fact("gt-kart", "Güvenlik", "Giriş Kartı",
         "Kayıp giriş kartları 2 saat içinde güvenlik birimine bildirilmelidir. Yedek kart ücreti 150 TL'dir ve maaştan kesilmez, kasaya yatırılır.",
         (("Giriş kartımı kaybedersem ne kadar süre içinde haber vermeliyim?", "2 saat"), ("Yeni kart için ne kadar ödenir?", "150 TL"))),
    Fact("gt-ziyaretci", "Güvenlik", "Ziyaretçi Kabulü",
         "Ziyaretçiler resepsiyonda kimlik ibraz ederek kayıt yaptırır ve her zaman bir çalışan eşliğinde dolaşır. Ziyaretçi kartı gün sonunda iade edilir.",
         (("Dışarıdan gelen misafir binada tek başına dolaşabilir mi?", "çalışan eşliğinde"), ("Konuk kartı ne zaman geri verilir?", "gün sonunda"))),
    Fact("gt-otopark", "Güvenlik", "Otopark Kullanımı",
         "Otopark tahsisi 3 ayda bir yapılan kurayla yapılır ve engelli çalışanlara öncelik verilir. Aracını kullanmayan çalışanın hakkı 30 gün sonra iptal edilir.",
         (("Park yeri kaç ayda bir dağıtılır?", "3 ayda bir"), ("Kullanılmayan park hakkı ne zaman düşer?", "30 gün"))),
    Fact("gt-yangin", "Güvenlik", "Yangın Tatbikatı",
         "Binalarda yılda iki kez tahliye tatbikatı yapılır ve katılım zorunludur. Toplanma alanları kat planlarında belirtilmiştir.",
         (("Tahliye tatbikatı yılda kaç kez yapılır?", "iki kez"), ("Acil durumda nerede toplanmalıyım?", "toplanma alanları"))),
    Fact("gt-masa", "Güvenlik", "Temiz Masa Politikası",
         "Mesai bitiminde gizli belgeler kilitli dolaplarda saklanmalı ve ekranlar kilitlenmelidir. Denetimler ayda bir yapılır ve ihlaller yöneticiye raporlanır.",
         (("İş çıkışında masada evrak bırakmak yasak mı?", "kilitli dolaplarda"), ("Masa denetimleri hangi sıklıkta?", "ayda bir"))),
    Fact("gt-isg", "Güvenlik", "İş Kazası Bildirimi",
         "İş kazaları olay anından itibaren 24 saat içinde İSG birimine bildirilir. Yaralanma gerektiren kazalar için olay yeri fotoğraflanır.",
         (("İşte yaralanırsam ne kadar sürede bildirim yapılmalı?", "24 saat"), ("Kaza sonrası olay yerinde ne yapılır?", "fotoğraflanır"))),
    # --- İletişim ve Satış
    Fact("ki-sosyal", "İletişim", "Sosyal Medya Kullanımı",
         "Çalışanlar şirket adına açıklama yapamaz; basın talepleri kurumsal iletişim birimine yönlendirilir. Kişisel hesaplarda müşteri bilgileri paylaşılamaz.",
         (("Gazeteci bana soru sorarsa ne yapmalıyım?", "kurumsal iletişim birimine"), ("Kişisel Instagram hesabımda hangi bilgi paylaşılamaz?", "müşteri bilgileri"))),
    Fact("ki-marka", "İletişim", "Marka Kullanımı",
         "Logo ve kurumsal renkler marka rehberine uygun kullanılmalıdır. Dış sunumlar yayımlanmadan önce kurumsal iletişim onayından geçer.",
         (("Dışarıya yapılacak sunumlar yayından önce kime gösterilir?", "kurumsal iletişim onayı"), ("Logo kullanımı için hangi doküman geçerli?", "marka rehberi"))),
    Fact("sl-indirim", "Satış", "Satış İndirim Yetkileri",
         "Satış temsilcileri %5'e, bölge müdürleri %12'ye kadar indirim verebilir. %12 üzerindeki indirimler satış direktörünün onayına tabidir.",
         (("Satış elemanı en fazla ne kadar indirim yapabilir?", "%5"), ("%15'lik indirim için kim onaylar?", "satış direktörü"))),
    Fact("sl-tahsilat", "Satış", "Tahsilat Takibi",
         "Vadesi 30 gün geçen alacaklar için müşteriye hatırlatma gönderilir. 90 günü geçen alacaklar hukuk takibine devredilir.",
         (("Ödemesi geciken müşteriye ne zaman uyarı yapılır?", "30 gün"), ("Alacak kaç gün gecikirse hukuka aktarılır?", "90 gün"))),
    Fact("sl-teklif", "Satış", "Müşteri Teklif Geçerliliği",
         "Müşteri teklifleri düzenlendiği tarihten itibaren 30 gün geçerlidir. Geçerlilik uzatma talepleri bölge müdürü onayıyla yapılır.",
         (("Hazırladığım fiyat teklifi ne kadar süre geçerli?", "30 gün"), ("Teklifin süresini uzatmak istersem kime başvururum?", "bölge müdürü"))),
    Fact("sl-sikayet", "Satış", "Müşteri Şikâyet Süresi",
         "Müşteri şikâyetlerine ilk yanıt 24 saat içinde verilir ve çözüm 10 iş günü içinde tamamlanır.",
         (("Şikâyet eden müşteriye ilk cevap ne zaman verilmeli?", "24 saat"), ("Sorunun kapatılması için tanınan süre nedir?", "10 iş günü"))),
]

# Topics that are genuinely absent from the knowledge base. Several share words with it
# ("izin", "bütçe", "sağlık") on purpose, so abstention cannot rely on vocabulary alone.
UNANSWERABLE: list[str] = [
    "Yıllık izin hakkım kaç gün?",
    "Yemekhanede bugün hangi yemek çıkıyor?",
    "Şirket servis araçlarının güzergâhları neler?",
    "Çalışanlara hisse senedi opsiyonu veriliyor mu?",
    "Doğum izni kaç hafta sürüyor?",
    "Maaş zamları ne zaman yapılıyor?",
    "Özel sağlık sigortası diş tedavisini karşılıyor mu?",
    "Kripto para ile ödeme kabul ediyor muyuz?",
    "Yabancı dil kursu desteği var mı?",
    "Şirketin hisse fiyatı bugün ne kadar?",
    "Evcil hayvanımı ofise getirebilir miyim?",
    "Uzaktan çalışma günlerinde ev ofis desteği ödeniyor mu?",
]
