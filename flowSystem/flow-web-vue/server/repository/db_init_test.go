package repo

import (
	model "flow-server/models"
	"fmt"
	"os"
	"testing"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

var tables = struct {
	UserBasic string
}{
	UserBasic: "user_basic",
}

var db_ *gorm.DB

func init() {
	var err error
	db_, err = gorm.Open(mysql.Open("root:fuck@tcp(127.0.0.1:3306)/flow_system?charset=utf8mb4&parseTime=True&loc=Local"), &gorm.Config{})
	if err != nil {
		panic("Failed to connect to database: %v" + err.Error())
	}
}

func TestCreateUser(t *testing.T) {
	db_.AutoMigrate(&model.User{})

	// db_.Create(&model.User{
	// 	Email:    "test@example.com",
	// 	Password: "123456",
	// 	Nickname: "Test User",
	// 	Avatar:   "https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png",
	// 	Identity: 1,
	// })

	// db_.Where("email = ?", "test@example.com").Updates(&model.User{
	// 	Email:    "test@example.com",
	// 	Password: "123456",
	// 	Nickname: "元始天尊",
	// 	Avatar:   "https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png",
	// 	Identity: 1,
	// 	Username: "Sora",
	// })
}
func TestDeleteUser(t *testing.T) {
	// 删除数据表
	db_.Exec("DROP TABLE IF EXISTS user")
}

func TestCreateFlowPattern(t *testing.T) {
	db_.AutoMigrate(&model.FlowPattern{})

	db_.Unscoped().Create(&model.FlowPattern{
		PatternType:    "段塞流",
		PredictedLabel: 0,
		Prob:           0.9,
		Status:         1,
		Data:           []float64{1.0, 2.0, 3.0, 4.0, 5.2, 1.24648213856},
		Addr:           "检测地点A",
	})

	// db_.Where("id = ?", 1).Updates(&model.FlowPattern{
	// 	ID:             1,
	// 	PatternType:    "段塞流",
	// 	PredictedLabel: 1,
	// 	Prob:           0.9,
	// 	Status:         1,
	// 	Data:           []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1254},
	// 	Addr:           "检测地点A",
	// })
	// var flowPattern model.FlowPattern
	// tx := db_.Where("id = ?", 1).First(&flowPattern)
	// if tx.Error != nil {
	// 	t.Errorf("Failed to fetch flow pattern: %v", tx.Error)
	// }
	// fmt.Println(flowPattern)
	// fmt.Printf("%T\n", flowPattern.Data)
}

func Test22(t *testing.T) {
	file, err := os.Open("../upload_data/flow/pressure/6/G231L76.txt")
	if err != nil {
		t.Errorf("Failed to open file: %v", err)
		return
	}
	defer file.Close()
	for range 3 {
		var data []float64

		// 4096行数据
		for range 4096 {
			var line float64
			_, err := fmt.Fscanf(file, "%f\n", &line)
			if err != nil {
				t.Errorf("Failed to read line: %v", err)
				return
			}
			data = append(data, line)
		}

		db_.Unscoped().Create(&model.FlowPattern{
			PatternType:    "泡沫环状流",
			PredictedLabel: 6,
			Prob:           0.999,
			Status:         1,
			Data:           data,
			Addr:           "检测地点E",
		})
	}
}

func TestCreateDeviceAddress(t *testing.T) {
	db_.AutoMigrate(&model.DeviceAddress{})

	db_.Unscoped().Create(&model.DeviceAddress{
		Name:       "检测地点A",
		Location:   "详细位置A",
		DeviceCode: "设备编号A",
		DeviceType: 4,
		Status:     1,
		IP:         "192.168.1.1",
		Port:       5000,
		AutoDetect: true,
	})
}
