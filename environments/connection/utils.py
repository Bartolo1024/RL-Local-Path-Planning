import rospy


def wait_for_service(service_name, try_function):
    rospy.wait_for_service(service_name)
    try:
        try_function()
    except (rospy.ServiceException) as e:
        print(e)
        print('{} service call failed'.format(service_name))


def wait_for_data(service_name, service_lib, time_out):
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message(service_name, service_lib, timeout=time_out)
        except:
            pass
    return data