package repo

import model "flow-server/models"

func FetchDeviceAddressList(keyword string, status int, pageNum int, pageSize int) ([]model.DeviceAddress, int64, error) {
	var deviceAddrList []model.DeviceAddress
	var total int64

	tx := DB.Model(&model.DeviceAddress{})
	if keyword != "" {
		tx = tx.Where("name LIKE ? OR location LIKE ? OR device_code LIKE ?", "%"+keyword+"%", "%"+keyword+"%", "%"+keyword+"%")
	}
	if status != -1 {
		tx = tx.Where("status = ?", status)
	}

	tx.Count(&total)
	err := tx.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&deviceAddrList).Error
	if err != nil {
		return nil, 0, err
	}
	return deviceAddrList, total, nil
}

func AddDeviceAddress(deviceAddr model.DeviceAddress) error {
	return DB.Unscoped().Create(&deviceAddr).Error
}

func GetDeviceAddress(id uint) (*model.DeviceAddress, error) {
	var deviceAddr model.DeviceAddress
	err := DB.Where("id = ?", id).First(&deviceAddr).Error
	if err != nil {
		return nil, err
	}
	return &deviceAddr, nil
}

func UpdateDeviceAddress(deviceAddr *model.DeviceAddress) error {
	return DB.Save(deviceAddr).Error
}

func DeleteDeviceAddress(id uint) error {
	return DB.Where("id = ?", id).Delete(&model.DeviceAddress{}).Error
}

func BatchDeleteDeviceAddress(ids *[]uint) error {
	return DB.Delete(&model.DeviceAddress{}, "id IN ?", *ids).Error
}
