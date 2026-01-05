package service

import (
	model "flow-server/models"
	repo "flow-server/repository"
	"flow-server/resp"

	"github.com/gin-gonic/gin"
)

type DeviceAddressListRequest struct {
	Keyword  string
	Status   int
	PageNum  int
	PageSize int
}

func GetDeviceAddressList(c *gin.Context) {
	var req DeviceAddressListRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "参数绑定失败")
		return
	}
	deviceAddrList, total, err := repo.FetchDeviceAddressList(req.Keyword, req.Status, req.PageNum, req.PageSize)
	if err != nil {
		resp.FailWithMsg(c, "查询设备地址列表失败")
		return
	}
	resp.OKWithData(c, gin.H{
		"device_addr_list": deviceAddrList,
		"total":            total,
	})
}

func AddDeviceAddress(c *gin.Context) {
	var req model.DeviceAddress
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "参数绑定失败")
		return
	}
	err = repo.AddDeviceAddress(req)
	if err != nil {
		resp.FailWithMsg(c, "新增设备地址失败")
		return
	}
	resp.OKWithMsg(c, "新增设备地址成功")
}

func UpdateDeviceAddress(c *gin.Context) {
	var req model.DeviceAddress
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "参数绑定失败")
		return
	}
	deviceAddr, err := repo.GetDeviceAddress(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "查询设备地址失败")
		return
	}

	deviceAddr.Name = req.Name
	deviceAddr.Location = req.Location
	deviceAddr.DeviceCode = req.DeviceCode
	deviceAddr.DeviceType = req.DeviceType
	deviceAddr.Status = req.Status
	deviceAddr.IP = req.IP
	deviceAddr.Port = req.Port
	deviceAddr.SampleLength = req.SampleLength
	deviceAddr.Step = req.Step
	deviceAddr.AutoDetect = req.AutoDetect
	err = repo.UpdateDeviceAddress(deviceAddr)
	if err != nil {
		resp.FailWithMsg(c, "更新设备地址失败")
		return
	}
	resp.OKWithMsg(c, "更新设备地址成功")
}

type DeleteDeviceAddressRequest struct {
	ID uint `json:"id"`
}

func DeleteDeviceAddress(c *gin.Context) {
	var req DeleteDeviceAddressRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "参数绑定失败")
		return
	}
	err = repo.DeleteDeviceAddress(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "删除设备地址失败")
		return
	}
	resp.OKWithMsg(c, "删除设备地址成功")
}

type BatchDeleteDeviceAddressRequest struct {
	IDs []uint `json:"ids"`
}

func BatchDeleteDeviceAddress(c *gin.Context) {
	var req BatchDeleteDeviceAddressRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "参数绑定失败")
		return
	}
	err = repo.BatchDeleteDeviceAddress(&req.IDs)
	if err != nil {
		resp.FailWithMsg(c, "批量删除设备地址失败")
		return
	}
	resp.OKWithMsg(c, "批量删除设备地址成功")
}

type UpdateAddressStatusRequest struct {
	ID     uint `json:"id"`
	Status int  `json:"status"`
}

func UpdateAddressStatus(c *gin.Context) {
	var req UpdateAddressStatusRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "参数绑定失败")
		return
	}
	deviceAddr, err := repo.GetDeviceAddress(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "查询设备地址失败")
		return
	}
	deviceAddr.Status = req.Status
	err = repo.UpdateDeviceAddress(deviceAddr)
	if err != nil {
		resp.FailWithMsg(c, "更新设备地址状态失败")
		return
	}
	resp.OKWithMsg(c, "更新设备地址状态成功")
}
