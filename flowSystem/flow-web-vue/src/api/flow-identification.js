import { post } from "@/assets/js/http";

export async function runFlowIdentification(address,data) {
  return post(address+'/api/predict', data);
}
