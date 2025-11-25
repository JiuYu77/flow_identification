package router

import (
	"flow-server/controller"
	"flow-server/middleware"

	"github.com/gin-gonic/gin"
)

func Router() *gin.Engine {
	r := gin.Default()
	r.Static("assets", "dist/assets")
	r.Static("public", "dist/public") // 提供 public 目录下的静态文件
	r.Static("img", "dist/img")

	r.Static("upload_data", "upload_data") // upload_data 目录下的文件可以直接访问  上传的文件

	// 界面路由
	r.NoRoute(controller.NoRoute) // 处理所有其他路由，返回 index.html

	// API路由
	group := r.Group("/api")
	{
		// 公开接口（无需认证）
		publicUser := group.Group("/user")
		{
			publicUser.POST("/login", controller.Login)                          // 用户登录接口
			publicUser.POST("/register", controller.Register)                    // 用户注册接口
			publicUser.POST("/forgot-password", controller.SendVerificationCode) // 忘记密码，发送验证码接口
			publicUser.POST("/reset-password", controller.ResetPassword)         // 忘记密码，重置密码接口
		}

		// 受保护接口（需要登录）
		protectedUser := group.Group("/user")
		protectedUser.Use(middleware.AuthMiddleware())
		{
			protectedUser.POST("/logout", controller.Logout)                  // 用户登出接口
			protectedUser.GET("/me", controller.FetchUserInfo)                // 用户信息接口
			protectedUser.POST("/update", controller.UpdateUserInfo)          // 用户更新接口
			protectedUser.POST("/list", controller.FetchUserList)             // 用户列表接口
			protectedUser.POST("/delete", controller.DeleteUser)              // 用户删除接口
			protectedUser.GET("/profile", controller.GetCurrentUserProfile)   // 获取个人信息接口
			protectedUser.POST("/profile/update", controller.UpdateUserInfo)  // 更新个人信息接口
			protectedUser.POST("/change-password", controller.ChangePassword) // 修改密码接口
			protectedUser.POST("/update-avatar", controller.UpdateUserAvatar) // 更新头像接口
		}

		auth := group.Group("/auth")
		{
			auth.POST("/validate", controller.Validate) // 用户验证接口
			auth.POST("/refresh", controller.Refresh)   // 用户刷新接口
		}

		// 流型分析接口
		flowPattern := group.Group("/flow-pattern")
		flowPattern.Use(middleware.AuthMiddleware())
		{
			flowPattern.POST("/list", controller.GetFlowPatternList)             // 流型分析列表接口
			flowPattern.POST("/update", controller.UpdateFlowPattern)            // 流型分析更新接口
			flowPattern.POST("/delete", controller.DeleteFlowPattern)            // 流型分析删除接口
			flowPattern.POST("/batch-delete", controller.BatchDeleteFlowPattern) // 流型分析批量删除接口
			flowPattern.POST("/create", controller.CreateFlowPattern)            // 流型分析创建接口
		}

		// 设备地址接口
		deviceAddress := group.Group("/device/address")
		deviceAddress.Use(middleware.AuthMiddleware())
		{
			deviceAddress.POST("/list", controller.GetDeviceAddressList)             // 设备地址列表接口
			deviceAddress.POST("/add", controller.AddDeviceAddress)                  // 新增设备地址接口
			deviceAddress.POST("/update", controller.UpdateDeviceAddress)            // 更新设备地址接口
			deviceAddress.POST("/delete", controller.DeleteDeviceAddress)            // 删除设备地址接口
			deviceAddress.POST("/batch-delete", controller.BatchDeleteDeviceAddress) // 批量删除设备地址接口
			deviceAddress.POST("/update-status", controller.UpdateAddressStatus)     // 更新地址状态接口
		}

		// 数据文件接口
		file := group.Group("/data-file")
		file.Use(middleware.AuthMiddleware())
		{
			file.POST("/list", controller.GetDataFileList)             // 数据文件列表接口
			file.GET("/tree", controller.GetDataFileTree)              // 数据文件树接口
			file.POST("/create-dir", controller.CreateDataDir)         // 创建数据目录接口
			file.POST("/delete", controller.DeleteDataFile)            // 删除数据文件或目录接口
			file.POST("/upload", controller.UploadDataFile)            // 上传数据文件接口
			file.POST("/download", controller.DownloadDataFile)        // 下载数据文件接口
			file.POST("/batch-delete", controller.BatchDeleteDataFile) // 批量删除数据文件接口
		}
		file2 := group.Group("/data-file")
		{
			file2.GET("/download/:token", controller.DownloadFileByToken) // 下载数据文件接口
		}
	}

	return r
}
