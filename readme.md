# How to run

yang bener pakai fix.ipynb, itu tinggal jalanin aja, nanti dia download bert kurleb 400MB an, trs bikin model, kalau mas nya mau di bagian `for epoch in range` nah itu 512 bisa di ganti kelipatan 2 nya, itu kyk ngulangin proses training, tapi yang ideal epoch nya ngga usah lama2 tapi loss nya langsung turun drastis

trs model di simpen di file `improved_model` trs di server pakai flask, install dulu `pip install -r requirements.txt` trs python server : `python serve.py`
