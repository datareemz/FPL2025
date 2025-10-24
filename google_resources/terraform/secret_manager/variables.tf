variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "secret_id" {
  type        = string
  description = "The ID of the secret to create."
}

variable "secret_data" {
  type        = string
  description = "The secret data to store (email and password in JSON format)."
  sensitive   = true
}

variable "replication_auto" {
  type        = bool
  description = "Enable automatic replication across all regions."
  default     = true
}