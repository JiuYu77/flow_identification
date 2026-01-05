package resp

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type Response struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data any    `json:"data"`
}

var (
	CodeMap = map[int]string{
		0:    "success",
		1000: "服务错误",
		1001: "", // 传递 msg  使用
		1002: "需要登录",
		1003: "权限错误",
		1004: "角色错误",
	}
	DefaultMsg = CodeMap[1000]
)

func response(c *gin.Context, code int, msg string, data any) {
	c.JSON(http.StatusOK, Response{
		Code: code,
		Msg:  msg,
		Data: data,
	})
}

func OK(c *gin.Context, msg string, data any) {
	response(c, 0, msg, data)
}
func OKWithMsg(c *gin.Context, msg string) {
	OK(c, msg, gin.H{})
}
func OKWithData(c *gin.Context, data any) {
	OK(c, "成功", data)
}

func Fail(c *gin.Context, code int, msg string, data any) {
	response(c, code, msg, data)
}
func FailWithMsg(c *gin.Context, msg string) {
	Fail(c, 1001, msg, gin.H{})
}
func FailWithCode(c *gin.Context, code int) {
	msg, ok := CodeMap[code]
	if !ok {
		msg = DefaultMsg
	}
	Fail(c, code, msg, gin.H{})
}
