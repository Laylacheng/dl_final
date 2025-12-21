# dl_final
這份專案是Dual Encoder架構，使用image encoder跟text encoder分別抓影像與語言的語義特徵。透過projection layer將兩個特徵映射到共享空間，之後用CLIP Loss進行對比式學習以及使用矩陣運算計算相似度，最後評估就用NDCG@5 指標衡量檢索結果的準確性與排序品質。

requirements.txt 放相關需要的套件。

projection_heads.pth 這是模型的資料，因為我們不需要模型在介面的時候還要重新訓練，他直接拿我們的.pth檔案再去評估就好。

test_caption_list.csv  測試集的候選資料。
