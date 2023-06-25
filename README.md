# xosomienbac_statistics_analysis
Thống kê và phân tích dữ liệu xổ số Miền Bắc, để biết rằng, việc kiếm tiền từ nó là khó như thế nào.


# Xổ Số Miền Bắc - Thống kê & Phân tích dữ liệu

Dự án này thực hiện việc phân tích và thống kê dữ liệu Xổ Số Miền Bắc, giúp người chơi hiểu rõ hơn về cuộc chơi, vị thế, xác suất trong cuộc chơi.

## Tính năng chính

1. Thu thập dữ liệu Xổ Số Miền Bắc.
2. Phân tích và thống kê dữ liệu.
3. Dự báo các con số có khả năng xuất hiện cao trong các kỳ quay số tiếp theo (Áp dụng mô hình học máy "Random Forest Regressor" để dự đoán dữ liệu).
4. Thống kê xác suất chiến thắng, số tiền vốn qua thời gian (từ 2003 đến 2023)

## Cài đặt

1. Clone dự án về máy tính của bạn:
    ```
    git clone https://github.com/hukhho/xosomienbac_statistics_analysis.git
    ```
2. Cài đặt môi trường và các thư viện cần thiết:
    ```
    cd xsmb_statistics_analysis
    pip install -r requirements.txt
    ```
3. Tạo database: 
Sử dụng MSSQL: ```CREATE DATABASE hunglotto``` 
Chạy file ```xosodata2002to2023.sql```
3. Tổng quan:
    Ở đây là chơi xổ số với rate return là 97 (theo một số trang hiện nay) với tổng con số chơi mỗi đợt là 51.

    Có thể sửa thuật toán quản lý vốn ở hàm "update_statistics()"
    Ở đây đang sử dụng phương pháp thua gấp thếp, đến 10 lần thì stoploss. Nếu sử dụng 15 lần gấp thếp thì tài sản sẽ không bao giờ lỗ.
    Vì trong quá khứ 20 năm từ 2002 đến 2023 chuỗi thua lớn nhất là 14. Nhưng số tiền lãi chỉ là 10% tổng số vốn 100.000 sau 20 năm lãi 10% thì tiền thua cả tiền lạm phát, chơi làm gì.

    Chương trình đang sử dụng dự đoán số theo mô hình RFR, sau đó lấy 50 số gần nhất với con số mô hình dự đoán. (Tổng 51 số)
    Có thể sửa trong "generate_predict_numbers.py" để cho ra thuật toán dự đoán chính xác với winrate cao hơn.

    Tổng quan lại, ĐỂ KIẾM TIỀN TỪ XỔ SỐ TRONG KHOẢNG THỜI GIAN 20 NĂM ỔN ĐỊNH gần như bất khả thi.~~


    ```
    python xoso.py
    ```

## Giấy phép

Dự án này được phân phối dưới Giấy phép MIT. Xem file [LICENSE](./LICENSE) để biết thêm chi tiết.

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc góp ý, vui lòng liên hệ với tôi qua email: hukhho@gmail.com
