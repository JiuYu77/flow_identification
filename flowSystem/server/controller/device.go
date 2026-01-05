package controller

import (
	"flow-server/service"

	"github.com/gin-gonic/gin"
)

func GetDeviceAddressList(c *gin.Context) {
	service.GetDeviceAddressList(c)
}

func AddDeviceAddress(c *gin.Context) {
	service.AddDeviceAddress(c)
}

func UpdateDeviceAddress(c *gin.Context) {
	service.UpdateDeviceAddress(c)
}
func DeleteDeviceAddress(c *gin.Context) {
	service.DeleteDeviceAddress(c)
}

func BatchDeleteDeviceAddress(c *gin.Context) {
	service.BatchDeleteDeviceAddress(c)
}

func UpdateAddressStatus(c *gin.Context) {
	service.UpdateAddressStatus(c)
}
