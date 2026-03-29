import { db } from './firebase-config';
import { collection, getDocs } from "firebase/firestore"; // Thêm dòng này

console.log("Tox-Agent đã sẵn sàng kết nối Firebase!");

// Hàm chạy thử để kiểm tra kết nối Database
async function testDatabase() {
  console.log("Đang kiểm tra kết nối Firestore...");
  // Thử lấy dữ liệu từ một collection tên là 'research' (nếu có)
  const querySnapshot = await getDocs(collection(db, "research"));
  console.log(`Đã kết nối thành công! Tìm thấy ${querySnapshot.size} tài liệu.`);
}

testDatabase(); // Gọi hàm chạy thử

document.querySelector('#app').innerHTML = `
  <h1>Tox-Agent Project</h1>
  <p>Hệ thống dự đoán độc tính thuốc đang được triển khai...</p>
`;