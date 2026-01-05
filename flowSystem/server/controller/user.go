package controller

import (
	"flow-server/service"

	"github.com/gin-gonic/gin"
)

func Login(c *gin.Context) {
	service.Login(c)
}

func Register(c *gin.Context) {
	service.Register(c)
}

func Logout(c *gin.Context) {
	service.Logout(c)
}

func Validate(c *gin.Context) {
	service.Validate(c)
}
func Refresh(c *gin.Context) {
	service.Refresh(c)
}
func FetchUserInfo(c *gin.Context) {
	service.FetchUserInfo(c)
}
func UpdateUserInfo(c *gin.Context) {
	service.UpdateUserInfo(c)
}

func FetchUserList(c *gin.Context) {
	service.FetchUserList(c)
}

func DeleteUser(c *gin.Context) {
	service.DeleteUser(c)
}

// 新增个人信息相关控制器函数
func GetCurrentUserProfile(c *gin.Context) {
	service.GetCurrentUserProfile(c)
}

func ChangePassword(c *gin.Context) {
	service.ChangePassword(c)
}

// 更新头像功能
func UpdateUserAvatar(c *gin.Context) {
	service.UpdateUserAvatar(c)
}

func SendVerificationCode(c *gin.Context) {
	service.SendVerificationCode(c)
}

func ResetPassword(c *gin.Context) {
	service.ResetPassword(c)
}
