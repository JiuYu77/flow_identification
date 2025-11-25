package model

import (
	"time"

	"gorm.io/gorm"
)

type DeviceAddress struct {
	ID        uint           `gorm:"primarykey" json:"id"`
	CreatedAt time.Time      `json:"createdAt"`
	UpdatedAt time.Time      `json:"updatedAt"`
	DeletedAt gorm.DeletedAt `gorm:"index" json:"deletedAt"`

	Name       string `json:"name"`       // 地址名称
	Location   string `json:"location"`   // 详细位置
	DeviceCode string `json:"deviceCode"` // 设备编号
	DeviceType int    `json:"deviceType"` // 设备类型
	Status     int    `json:"status"`     // 状态，0：禁用，1：正常、已启用，2：警告、已启用，3：错误、不可用
	IP         string `json:"ip"`         // IP 地址
	Port       int    `json:"port"`       // 端口号

	SampleLength int  `json:"sampleLength"` // 样本长度
	Step         int  `json:"step"`         // 步长
	AutoDetect   bool `json:"autoDetect"`   // 自动检测，true：是，false：否
}

func (DeviceAddress) TableName() string {
	return "device_address"
}
