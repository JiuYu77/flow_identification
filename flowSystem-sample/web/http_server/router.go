package http_server

func Router(addr string, opts ...string) *Http {
	h := NewHttp(addr, opts...)

	// css js 等静态文件
	h.Static("/assets/", "/assets", "assets")

	// 页面
	h.GET("/", Index)
	// 测试
	h.GET("/a", A)

	return h
}
