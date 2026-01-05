package repo

import (
	"context"
	"flow-server/config"
	"log"

	"github.com/redis/go-redis/v9"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var (
	DB          *gorm.DB
	RedisClient *redis.Client
)

func InitMySQL() {
	var err error
	DB, err = gorm.Open(
		mysql.Open(config.SysConfig.MySQL.DSN),
		&gorm.Config{
			Logger: logger.Default.LogMode(logger.Info),
		},
	)
	if err != nil {
		panic("fail to connect mysql database: " + err.Error())
	}
	log.Println("连接MySQL数据库成功.")
}

func InitRedis() {
	RedisClient = redis.NewClient(&redis.Options{
		Addr:         config.SysConfig.Redis.Addr,
		Password:     config.SysConfig.Redis.Password,
		DB:           config.SysConfig.Redis.DB,
		PoolSize:     config.SysConfig.Redis.PoolSize,
		MinIdleConns: config.SysConfig.Redis.MinIdleConns,
	})

	pong, err := RedisClient.Ping(context.Background()).Result()
	if err != nil {
		panic("fail to connect redis database: " + err.Error())
	}
	log.Println("连接Redis数据库成功. 回复:", pong)
}
