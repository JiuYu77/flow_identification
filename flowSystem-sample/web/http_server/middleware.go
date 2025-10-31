package http_server

import "net/http"

type Middleware func(http.HandlerFunc) http.HandlerFunc

func Method(method string) Middleware {
	return func(handler http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Method != method { // 检查请求方法是否合法
				w.WriteHeader(http.StatusMethodNotAllowed)
				LogNotAllowed(method, r.URL.Path) // log
				return
			}

			handler(w, r) // Call the next middleware/handler in chain
		}
	}
}

func ValidPath(path string) Middleware {
	return func(handler http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != path { // 检查请求路径是否合法
				http.NotFound(w, r)
				LogNotFound(r.Method, r.URL.Path) // log
				return
			}
			LogFound(r.Method, r.URL.Path) // log

			handler(w, r)
		}
	}
}

// chainHandlers 用于将多个中间件按顺序链接起来，形成一个处理链，最后调用handler
// 例如：chainHandlers(handler, m1, m2, m3) 等价于 m3(m2(m1(handler)))
func ChainHandlers(handler http.HandlerFunc, middlewares ...Middleware) http.HandlerFunc {
	for _, m := range middlewares {
		handler = m(handler)
	}
	return handler
}
