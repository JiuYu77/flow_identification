package repo

import (
	"fmt"
	"log"
	"testing"
)

var tablesMap = map[string]bool{
	tables.UserBasic: true,
}

// 定义结构体来接收表结构信息
type TableDesc struct {
	Field   string `gorm:"column:Field"`
	Type    string `gorm:"column:Type"`
	Null    string `gorm:"column:Null"`
	Key     string `gorm:"column:Key"`
	Default string `gorm:"column:Default"`
	Extra   string `gorm:"column:Extra"`
}

func TestShowTableStructure(t *testing.T) {
	ShowTableStructure(tables.UserBasic)
}

// ShowTableStructure 查看数据表结构
func ShowTableStructure(tableName string) {
	// ... 数据库连接代码 ...

	// 检查数据表是否存在
	if _, ok := tablesMap[tableName]; !ok {
		log.Fatalf("数据表 %s 不存在", tableName)
	}

	var desc []TableDesc

	// 执行DESC查询并将结果扫描到结构体切片中
	sql := fmt.Sprintf("DESC %s", tableName)
	err := db_.Raw(sql).Scan(&desc).Error
	if err != nil {
		log.Fatal(err)
	}

	// 输出表结构
	fmt.Println("表结构: " + tableName)
	fmt.Println("============================================================================")
	fmt.Printf("%-20s %-15s %-5s %-5s %-10s %-10s\n",
		"Field", "Type", "Null", "Key", "Default", "Extra")
	fmt.Println("----------------------------------------------------------------------------")

	for _, column := range desc {
		fmt.Printf("%-20s %-15s %-5s %-5s %-10s %-10s\n",
			column.Field,
			column.Type,
			column.Null,
			column.Key,
			column.Default,
			column.Extra)
	}
	fmt.Println("============================================================================")
}
