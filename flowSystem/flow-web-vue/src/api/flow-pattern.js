import { post, requestInterceptor } from '@/assets/js/http';

export async function getFlowPatternList(data) {
  return post('/api/flow-pattern/list', data, requestInterceptor());
}

export async function updateFlowPattern(data){
  return post('/api/flow-pattern/update', data, requestInterceptor())
}

export async function deleteFlowPattern(data){
  return post('/api/flow-pattern/delete', data, requestInterceptor())
}

export async function batchDeleteFlowPattern(data){
  return post('/api/flow-pattern/batch-delete', data, requestInterceptor())
}

export async function createFlowPattern(data){
  return post('/api/flow-pattern/create', data, requestInterceptor())
}
