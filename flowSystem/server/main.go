package main

import (
	"flow-server/config"
	"flow-server/controller"
	"flow-server/jwt"
	model "flow-server/models"
	repo "flow-server/repository"
	"flow-server/router"
	"flow-server/service"

	"flow-server/utils"
)

func main() {
	config.InitConfig()
	repo.InitMySQL()
	repo.InitRedis()
	utils.InitVerfier()
	jwt.InitJWT()
	controller.InitDataFile()

	r := router.Router()
	r.Run(config.SysConfig.Server.Addr)
}

func testUser() {
	password := "12345678"
	var err error
	password, err = service.HashPassword(password)
	if err != nil {
		panic("Failed to hash password: %v" + err.Error())
	}

	repo.DB.Where("email = ?", "test@example.com").Updates(&model.User{
		Email:    "test@example.com",
		Password: password,
		Nickname: "元始天尊",
		Avatar:   "https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png",
		Identity: 1,
		Username: "Sora",
	})
}
