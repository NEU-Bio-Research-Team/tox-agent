// src/main.js
import { db } from './firebase-config'; // Đảm bảo bạn đã có file này
console.log("Tox-Agent đã sẵn sàng kết nối Firebase!");

document.querySelector('#app').innerHTML = `
  <h1>Tox-Agent Project</h1>
  <p>Hệ thống dự đoán độc tính thuốc đang được triển khai...</p>
`;